"""joint.py -- fits in which samples SHARE parameters, or a parameter is free in some samples only.

`fit` (independent.py) solves every sample alone: a vmap of independent Levenberg-Marquardt fits. That
cannot express a property the samples have in common -- a relaxation factor of a common solvent, a field
error of the scanner, the magnetisation the sequence starts from -- nor a parameter only some samples have,
such as a second pool in two of six tubes. `fit_joint` can. Each FreeParam says whether it is one value per
sample (scope="sample", as in `fit`) or one value for all of them (scope="shared"), and in which samples it
is free at all (`samples`). Wherever a parameter is not free, the sample keeps the value its params carry,
exactly.

HOW IT STAYS CHEAP. It is one least-squares problem over every free value, but a sample's residual depends
only on its own values and the shared ones. So the Jacobian is built per sample, for that sample's slots
only -- one forward tangent per slot, not one per joint column -- and then scattered into the joint matrix.

THE OPTIMISER is `levenberg_marquardt`'s damped Gauss-Newton (lam x0.5 on accept, x3 on reject, the step
clipped to the bounds). Without a shared parameter the problem is block diagonal, and every sample gets its
own damping: that IS `fit`'s algorithm, and with a plain spec the two give the same numbers. With a shared
parameter the samples are coupled, and one damping governs the joint step.

`max_chunk` builds the Jacobian `max_chunk` samples per device call. It bounds memory, and it is the
workaround for batched GPU Jacobians that came back NaN for some batch shapes (seen on an RTX 3070 with
jax 0.6.2 and 0.7.2). In a coupled fit a step whose Jacobian has NaN or inf is rejected and counted.

The joint matrix is dense, (points x free values). That suits a phantom -- tens of samples, a few shared
values -- not ten thousand voxels with only per-voxel parameters, which is what `fit` is for.
"""
import numpy as np
import jax
import jax.numpy as jnp

from ..simulation.simulate import simulate_spectrum
from ..simulation.voxels import tile_params
from .lm import levenberg_marquardt
from .result import FitResult


def _layout(spec, n):
    """Where every free value lives.

    Returns free (n, L): slot l is free in sample i; G (n, L, P): joint column p feeds slot l of sample i;
    col_slot (P,): each joint column's slot; col_who: each joint column's samples.
    """
    free = np.zeros((n, spec.n), bool)
    cols = []
    for l, s in enumerate(spec.slots):
        who = list(range(n)) if s.samples is None else list(s.samples)
        missing = [i for i in who if not 0 <= i < n]
        if missing:
            raise ValueError(f"{s.name}: samples {missing} do not exist (the fit has {n})")
        free[who, l] = True
        cols += [(l, who)] if s.scope == "shared" else [(l, [i]) for i in who]
    G = np.zeros((n, spec.n, len(cols)))
    for p, (l, who) in enumerate(cols):
        G[who, l, p] = 1.0
    return free, G, np.array([l for l, _ in cols], int), [who for _, who in cols]


def _carried(params, spec):
    """(n, L): the value every slot has in every sample's params."""
    out = []
    for s in spec.slots:
        a = jnp.asarray(getattr(params, s.field))
        out.append(a if s.index is None else a[:, s.index])
    return jnp.stack(out, axis=1)


def _write(p, vals, spec):
    """One sample's params with every slot set to `vals` (physical units)."""
    upd = {}
    for l, s in enumerate(spec.slots):
        if s.index is None:
            upd[s.field] = vals[l]
        else:
            upd[s.field] = upd.get(s.field, getattr(p, s.field)).at[s.index].set(vals[l])
    return p._replace(**upd)


def _cov_to_sd_corr(cov, scale):
    """Covariance in scaled units -> standard deviations (physical) and correlations."""
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
    return sd * scale, corr


def fit_joint(experiment, spec, z_meas, n_iter=30, x0_phys=None, n_per_block=None, max_chunk=None,
              lam0=1e-2, callback=None):
    """Fit `spec`'s parameters of every sample at once, with shared and partly free parameters allowed.

    Parameters
    ----------
    experiment
        An :class:`~bmc_jax.fitting.experiment.Experiment` whose params carry one entry per sample
        (or one sample, used for all of them). `Experiment.stack` for a ladder.
    spec
        Which parameters are free. A FreeParam with `scope="shared"` is one value for every sample it is
        free in; `samples=(...)` frees it only there, and every other sample keeps the value its params
        carry.
    z_meas
        Measured Z-spectra, shape (n_samples, n_offsets).
    n_iter
        Levenberg-Marquardt steps (all of them run; there is no convergence test).
    x0_phys
        Starting values, physical units, shape (n_samples, n_parameters); default each slot's `x0`. A
        shared value starts at the entry of the first sample it is free in. Entries that are not free
        are ignored.
    n_per_block
        Offsets per experiment, so the result can report the residual per rung of a ladder.
    max_chunk
        Samples per device call when building the Jacobian (None = all at once).
    lam0
        Starting damping.
    callback
        Called as `callback(step, cost)` after every step of a coupled fit (progress of a long run).

    Returns
    -------
    FitResult
        `x` holds every sample's value of every slot: fitted where free, carried where not (`free`
        says which). `sd` and `corr` are the misfit-scaled error bars and correlations at the
        solution, `x0` the starts (see `unmoved`), `cost` the sum of squares after every step.

    Notes
    -----
    Without a shared parameter the samples are independent and each gets its own damping, which is
    `fit`'s algorithm; with a plain spec (every parameter per sample, free everywhere) the result equals
    `fit`'s. With a shared parameter there is one joint step and one damping.

    Error bars come from the Jacobian at the solution scaled by the misfit, s^2 = cost / (points - free
    values) -- per sample when the samples are independent, over the whole fit when they are coupled.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> spec = FitSpec([FreeParam("k", "cest_k", 0, 100.0, 5.0, 2000.0, 300.0),
    ...                 FreeParam("f", "cest_f", 0, 0.001, 1e-5, 0.03, 0.005),
    ...                 FreeParam("r2w", "water_r2", None, 1.0, 0.1, 5.0, 0.6, scope="shared"),
    ...                 FreeParam("f2", "cest_f", 1, 0.001, 0.0, 0.03, 0.002, samples=(0, 1))])
    >>> res = fit_joint(Experiment.stack(ladder), spec, Z, n_per_block=len(ppm))
    >>> res["r2w"], res.sd[:, 2]          # one value, repeated for every sample
    """
    z = jnp.atleast_2d(jnp.asarray(z_meas))
    n = z.shape[0]
    params = experiment.params
    if jnp.asarray(params.water_r1).ndim == 0:
        params = tile_params(params, n)
    if jnp.asarray(params.water_r1).shape[0] != n:
        raise ValueError(f"params carry {jnp.asarray(params.water_r1).shape[0]} samples, z_meas {n}")
    m0 = jnp.asarray(experiment.m0)
    if m0.ndim == 1:
        m0 = jnp.broadcast_to(m0, (n,) + m0.shape)
    dt = params.water_r1.dtype
    z = z.astype(dt)
    events, mz = experiment.events, experiment.mz_loc

    free, G, col_slot, col_who = _layout(spec, n)
    P = G.shape[2]
    scales = np.array([s.scale for s in spec.slots])
    carried = _carried(params, spec).astype(dt)
    X0 = (np.broadcast_to(spec.x0_phys, free.shape) if x0_phys is None else np.asarray(x0_phys, float))
    if X0.shape != free.shape:
        raise ValueError(f"x0_phys has shape {X0.shape}, expected {free.shape} (samples, parameters)")
    X0 = np.where(free, X0, np.asarray(carried, float))
    fr = jnp.asarray(free)
    spans = [(a, min(a + max_chunk, n)) for a in range(0, n, max_chunk)] if max_chunk else [(0, n)]
    cut = lambda tree, a, b: jax.tree.map(lambda v: v[a:b], tree)

    def model(loc, p, m0_i, free_i, car_i):          # one sample: its scaled slot values -> Z
        phys = (loc * spec.scales).astype(dt)
        vals = jnp.where(free_i, phys, car_i)
        return simulate_spectrum(_write(p, vals, spec), events, m0_i, return_full_trajectory=False)[:, mz]

    vz = jax.jit(jax.vmap(model))
    vjl = jax.jit(jax.vmap(jax.jacfwd(model)))      # (samples, points, slots): per sample, its own slots

    def per_sample(fn, loc):
        return jnp.concatenate([fn(loc[a:b], cut(params, a, b), m0[a:b], fr[a:b], carried[a:b])
                                for a, b in spans])

    nan_steps = 0
    if not any(s.scope == "shared" for s in spec.slots):
        # block diagonal: every sample alone, with its own damping -- fit's algorithm
        hold = None if spec.independent else fr.astype(dt)
        lo = jnp.asarray(spec.lo_phys / scales, dt)
        hi = jnp.asarray(spec.hi_phys / scales, dt)

        def solve(p, m0_i, f_i, c_i, z_i, x0_i, h_i):
            r = lambda x: model(x, p, m0_i, f_i, c_i) - z_i
            return levenberg_marquardt(r, x0_i, lo, hi, n_iter=n_iter, lam0=lam0, use_fwd=True,
                                       return_trajectory=True, free=h_i)
        vsolve = jax.vmap(solve, in_axes=(0, 0, 0, 0, 0, 0, None if hold is None else 0))
        X0s = jnp.asarray(X0 / scales, dt)
        outs = [vsolve(cut(params, a, b), m0[a:b], fr[a:b], carried[a:b], z[a:b], X0s[a:b],
                       None if hold is None else hold[a:b]) for a, b in spans]
        loc = jnp.concatenate([o[0] for o in outs])
        cost = np.asarray(jnp.concatenate([o[2] for o in outs]), float).sum(axis=0)
    else:
        # coupled: one joint step over every free value, one damping
        Gj = jnp.asarray(G, dt)
        locs = lambda x: jnp.einsum("ilp,p->il", Gj, x)
        col_scale = scales[col_slot]
        lo = jnp.asarray(spec.lo_phys[col_slot] / col_scale, dt)
        hi = jnp.asarray(spec.hi_phys[col_slot] / col_scale, dt)
        eye = jnp.eye(P, dtype=dt)

        def resid(x):
            return (jax.vmap(model)(locs(x), params, m0, fr, carried) - z).reshape(-1)

        def jac(x):
            return jnp.einsum("inl,ilp->inp", per_sample(vjl, locs(x)), Gj).reshape(-1, P)

        @jax.jit
        def step(x, lam, J, r):
            c = r @ r
            g, H = J.T @ r, J.T @ J
            xn = jnp.clip(x + jnp.linalg.solve(H + lam * (jnp.diag(jnp.diag(H)) + 1e-9 * eye), -g), lo, hi)
            cn = jnp.sum(resid(xn) ** 2)
            better = cn < c
            return jnp.where(better, xn, x), jnp.where(better, lam * 0.5, lam * 3.0), jnp.where(better, cn, c)

        resid_j = jax.jit(resid)
        x = jnp.asarray([X0[who[0], l] / scales[l] for l, who in zip(col_slot, col_who)], dt)
        lam, cost = jnp.asarray(lam0, dt), []
        for it in range(n_iter):
            r, J = resid_j(x), jac(x)
            if bool(jnp.all(jnp.isfinite(J))):
                x, lam, c = step(x, lam, J, r)
                cost.append(float(c))
            else:                                      # a NaN Jacobian: reject, damp harder, count it
                nan_steps += 1
                lam = lam * 3.0
                cost.append(float(r @ r))
            if callback is not None:
                callback(it, cost[-1])
        loc = locs(x)
        cost = np.asarray(cost)

    X = np.where(free, np.asarray(loc, float) * scales, np.asarray(carried, float))
    z_fit = np.asarray(per_sample(vz, loc), float)
    r_all = z_fit - np.asarray(z, float)

    # error bars and correlations at the solution, misfit-scaled
    sd = np.full(free.shape, np.nan)
    corr = np.full(free.shape + (spec.n,), np.nan)
    Jl = np.asarray(per_sample(vjl, loc), float)        # (samples, points, slots)
    if not any(s.scope == "shared" for s in spec.slots):
        for i in range(n):
            use = np.flatnonzero(free[i])
            Ju = Jl[i][:, use]
            s2 = float(r_all[i] @ r_all[i]) / max(r_all.shape[1] - use.size, 1)
            sd[i, use], corr[i][np.ix_(use, use)] = _cov_to_sd_corr(np.linalg.pinv(Ju.T @ Ju) * s2, scales[use])
    else:
        J = np.einsum("inl,ilp->inp", Jl, G).reshape(-1, P)
        s2 = float(np.sum(r_all ** 2)) / max(r_all.size - P, 1)
        sd_col, corr_col = _cov_to_sd_corr(np.linalg.pinv(J.T @ J) * s2, col_scale)
        for i in range(n):
            ps = [p for p in range(P) if i in col_who[p]]
            ls = col_slot[ps]
            sd[i, ls] = sd_col[ps]
            corr[i][np.ix_(ls, ls)] = corr_col[np.ix_(ps, ps)]
    return FitResult(X, z_fit, np.asarray(z, float), spec, n_per_block, free=free, sd=sd, corr=corr,
                     x0=X0, cost=cost, nan_steps=nan_steps)
