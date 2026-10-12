"""joint.py -- fits in which samples SHARE parameters, or a parameter is free in some samples only.

`fit` (independent.py) solves every sample alone: a vmap of independent Levenberg-Marquardt fits. That
cannot express a property the samples have in common -- a relaxation factor of a common solvent, a field
error of the scanner, the magnetisation the sequence starts from -- nor a parameter only some samples have,
such as a second pool in two of six tubes. `fit_joint` can. Each FreeParam says
  scope    one value per sample ("sample", as in `fit`), one for all of them ("shared"), or one per
           experiment of a stacked ladder, shared by the samples ("block": a drift between the rungs);
  samples  in which samples it is free at all; elsewhere the sample keeps what its params carry, exactly;
  mode     whether the number replaces the carried value ("value"), multiplies it ("factor": one R2 factor
           over per-sample measured R2) or shifts it ("add": a B0 offset over a measured B0 map).
The field "m_start" is the magnetisation the sequence starts from, as a fraction of equilibrium. Free as a
value, it rebuilds each sample's start state from its current pools at every step; as a factor, it scales
the start state the experiment carries.

THE B0 SPREAD INSIDE A SAMPLE. With `spread=B0Spread(...)` every sample is simulated as `spread.nodes`
isochromats around its own B0 and averaged (bmc_jax.simulation.spread). Its width is the field "b0_spread"
(Hz): free like any value -- usually one shared by every sample -- or, without a slot, held at
`spread.sigma_hz` (a number, or one per sample). Without `spread` nothing changes.

HOW IT STAYS CHEAP. It is one least-squares problem over every free value, but a sample's residual depends
only on its own values and the shared ones. So the Jacobian is built per sample, for that sample's slots
only -- one forward tangent per slot, not one per joint column -- and then scattered into the joint
matrix. A per-block slot costs one tangent per block by default (`block_jacobian="full"`). With
`block_jacobian="split"` it costs ONE: a rung's points depend only on that rung's value, so the derivative
along "every block at once" is split into its blocks' rows afterwards. Both are exact (checked against
`jax.jacfwd` of the whole residual in tests/test_joint.py); they differ in float32 rounding only, which is
enough to send a badly conditioned coupled fit along a different path to a different stall (seen on a
noise-free test: k 39.98 vs 42.15 at cost 1.7e-6 vs 2.2e-6). "full" keeps every earlier number.

THE OPTIMISER is `levenberg_marquardt`'s damped Gauss-Newton (lam x0.5 on accept, x3 on reject, the step
clipped to the bounds). Without a shared or per-block value the problem is block diagonal and every sample
gets its own damping: that IS `fit`'s algorithm, and with a plain spec the two give the same numbers. With
one, the samples are coupled and one damping governs the joint step.

`max_chunk` builds the Jacobian `max_chunk` samples per device call. It bounds memory, and it is the
workaround for batched GPU Jacobians that came back NaN for some batch shapes (seen on an RTX 3070 with
jax 0.6.2 and 0.7.2). In a coupled fit a step whose Jacobian has NaN or inf is rejected and counted.

The joint matrix is dense, (points x free values). That suits a phantom -- tens of samples, a few shared
values -- not ten thousand voxels with only per-voxel parameters, which is what `fit` is for.
"""
import numpy as np
import jax
import jax.numpy as jnp

from ..simulation.params import SimParams, initial_state
from ..simulation.simulate import simulate_spectrum
from ..simulation.spread import simulate_spectrum_spread
from ..simulation.voxels import tile_params
from .lm import levenberg_marquardt
from .result import FitResult

M_START = "m_start"
B0_SPREAD = "b0_spread"
PSEUDO = (M_START, B0_SPREAD)          # free values that are not SimParams fields


def _layout(spec, n, n_blocks):
    """Where every free value lives.

    A sample's local vector has one entry per slot, `n_blocks` for a per-block slot (from `pos`).
    Returns free (n, L); pos (L,); n_loc; G (n, n_loc, P): joint column p feeds local entry j of sample i;
    cols: (slot, samples, block or -1) per joint column.
    """
    width = [n_blocks if s.scope == "block" else 1 for s in spec.slots]
    pos = np.concatenate([[0], np.cumsum(width)[:-1]]).astype(int)
    free = np.zeros((n, spec.n), bool)
    cols = []
    for l, s in enumerate(spec.slots):
        who = list(range(n)) if s.samples is None else list(s.samples)
        missing = [i for i in who if not 0 <= i < n]
        if missing:
            raise ValueError(f"{s.name}: samples {missing} do not exist (the fit has {n})")
        free[who, l] = True
        if s.scope == "shared":
            cols.append((l, who, -1))
        elif s.scope == "block":
            cols += [(l, who, b) for b in range(n_blocks)]
        else:
            cols += [(l, [i], -1) for i in who]
    G = np.zeros((n, int(sum(width)), len(cols)))
    for p, (l, who, b) in enumerate(cols):
        G[who, pos[l] + max(b, 0), p] = 1.0
    return free, pos, int(sum(width)), G, cols


def _carried(params, m0, spec, sig_held=None):
    """(n, L): what each slot's number is where it is not free -- the field's value, 1 for a factor,
    0 for an offset, for m_start the start state's water Mz over its equilibrium, for b0_spread the held
    width."""
    out = []
    n = m0.shape[0]
    for s in spec.slots:
        if s.field == B0_SPREAD:
            out.append(jnp.asarray(sig_held, m0.dtype))
        elif s.mode == "factor":
            out.append(jnp.ones(n, m0.dtype))
        elif s.mode == "add":
            out.append(jnp.zeros(n, m0.dtype))
        elif s.field == M_START:
            out.append(m0[:, 2 * (1 + params.cest_k.shape[-1])] / jnp.asarray(params.water_f))
        else:
            a = jnp.asarray(getattr(params, s.field))
            out.append(a if s.index is None else a[:, s.index])
    return jnp.stack(out, axis=1)


def _apply(p, m0_i, eff, free_i, spec):
    """One sample's params and start state with every slot's number applied (physical units)."""
    upd, m_slot = {}, None
    for l, s in enumerate(spec.slots):
        if s.field == M_START:
            m_slot = l
            continue
        if s.field == B0_SPREAD:                       # not a SimParams field: the model reads it
            continue
        cur = upd.get(s.field, getattr(p, s.field))
        old = cur if s.index is None else cur[s.index]
        new = eff[l] if s.mode == "value" else old * eff[l] if s.mode == "factor" else old + eff[l]
        upd[s.field] = new if s.index is None else cur.at[s.index].set(new)
    q = p._replace(**upd)
    if m_slot is None:
        return q, m0_i
    m = eff[m_slot]
    rebuilt = initial_state(q) * m if spec.slots[m_slot].mode == "value" else m0_i * m
    return q, jnp.where(free_i[m_slot], rebuilt, m0_i)


def _cov_to_sd_corr(cov, scale):
    """Covariance in scaled units -> standard deviations (physical) and correlations."""
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
    return sd * scale, corr


def fit_joint(experiment, spec, z_meas, n_iter=30, x0_phys=None, n_per_block=None, max_chunk=None,
              lam0=1e-2, callback=None, spread=None, block_jacobian="full", _internals=False):
    """Fit `spec`'s parameters of every sample at once: shared, per-block and partly free values allowed.

    Parameters
    ----------
    experiment
        An :class:`~bmc_jax.fitting.experiment.Experiment` whose params carry one entry per sample
        (or one sample, used for all of them). `Experiment.stack` for a ladder.
    spec
        Which parameters are free and how (FreeParam `scope`, `samples`, `mode`; see the module notes).
    z_meas
        Measured Z-spectra, shape (n_samples, n_offsets).
    n_iter
        Levenberg-Marquardt steps (all of them run; there is no convergence test).
    x0_phys
        Starting values, physical units, shape (n_samples, n_parameters); default each slot's `x0`. A
        shared or per-block value starts at the entry of the first sample it is free in (every block
        alike). Entries that are not free are ignored.
    n_per_block
        Offsets per experiment of a stacked ladder: the residual per rung, and needed by per-block values.
    max_chunk
        Samples per device call when building the Jacobian (None = all at once).
    lam0
        Starting damping.
    callback
        Called as `callback(step, cost)` after every step of a coupled fit (progress of a long run).
    spread
        A :class:`~bmc_jax.simulation.spread.B0Spread`: simulate every sample as that many isochromats
        around its own B0 and average them. Its width is the slot "b0_spread" (Hz) where one is given,
        `spread.sigma_hz` elsewhere. None (default): one isochromat, as before.
    block_jacobian
        "full" (default): one forward tangent per block of a per-block slot, as before. "split": one per
        slot, split into its blocks' rows afterwards -- cheaper by the number of blocks, the same derivative
        up to rounding (see the module notes).
    _internals
        Tests only: return the residual, the Jacobian and the start instead of fitting.

    Returns
    -------
    FitResult
        `x` holds every sample's number for every slot: fitted where free, carried where not (`free` says
        which; for a factor the carried number is 1, for an offset 0). Per-block values are in `blocks`
        (`res[name]`), their column of `x` is NaN. `sd` and `corr` are misfit-scaled error bars and
        correlations at the solution, `x0` the starts (see `unmoved`), `cost` the sum of squares after
        every step.

    Notes
    -----
    Without a shared or per-block value the samples are independent and each gets its own damping, which
    is `fit`'s algorithm; with a plain spec (plain values per sample, free everywhere) the result equals
    `fit`'s. Otherwise there is one joint step and one damping.

    Error bars come from the Jacobian at the solution scaled by the misfit, s^2 = cost / (points - free
    values) -- per sample when the samples are independent, over the whole fit when they are coupled.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> spec = FitSpec([FreeParam("k", "cest_k", 0, 100.0, 5.0, 2000.0, 300.0),
    ...                 FreeParam("f", "cest_f", 0, 0.001, 1e-5, 0.03, 0.005),
    ...                 FreeParam("r2w", "water_r2", None, 1.0, 0.5, 1.5, 1.0, scope="shared", mode="factor"),
    ...                 FreeParam("m0", "m_start", None, 1.0, 0.1, 1.0, 0.6, scope="shared"),
    ...                 FreeParam("db0", "b0_inhom", None, 0.01, -0.1, 0.1, 0.0, scope="block", mode="add"),
    ...                 FreeParam("f2", "cest_f", 1, 0.001, 0.0, 0.03, 0.002, samples=(0, 1))])
    >>> res = fit_joint(Experiment.stack(ladder), spec, Z, n_per_block=len(ppm))
    >>> res["r2w"], res["db0"], res.sd
    """
    z = jnp.atleast_2d(jnp.asarray(z_meas))
    n, n_pts = z.shape
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
    unknown = [s.name for s in spec.slots if s.field not in PSEUDO and s.field not in SimParams._fields]
    if unknown:
        raise ValueError(f"{unknown}: not a SimParams field nor one of {PSEUDO}")
    l_sp = [l for l, s in enumerate(spec.slots) if s.field == B0_SPREAD]
    if l_sp and spread is None:
        raise ValueError(f"{[spec.slots[l].name for l in l_sp]}: b0_spread needs fit_joint(..., spread=B0Spread(...))")
    if len(l_sp) > 1:
        raise ValueError(f"{[spec.slots[l].name for l in l_sp]}: one b0_spread slot at most")
    l_sp = l_sp[0] if l_sp else None
    sig_held = None if spread is None else jnp.broadcast_to(jnp.asarray(spread.sigma_hz, dt), (n,))

    if block_jacobian not in ("full", "split"):
        raise ValueError(f"block_jacobian {block_jacobian!r}; use 'full' or 'split'")
    blocked = any(s.scope == "block" for s in spec.slots)
    if blocked and (not n_per_block or n_pts % n_per_block):
        raise ValueError(f"a per-block value needs n_per_block dividing the {n_pts} offsets (got {n_per_block})")
    nb = n_pts // n_per_block if blocked else 1
    free, pos, n_loc, G, cols = _layout(spec, n, nb)
    P = G.shape[2]
    L = spec.n
    scales = np.array([s.scale for s in spec.slots])
    is_block = np.array([s.scope == "block" for s in spec.slots])
    carried = _carried(params, m0, spec, sig_held).astype(dt)
    # what the model is handed per sample: the carried values, then (with a spread) the held width
    carried_in = carried if spread is None else jnp.concatenate([carried, sig_held[:, None]], axis=1)
    X0 = (np.broadcast_to(spec.x0_phys, free.shape) if x0_phys is None else np.asarray(x0_phys, float))
    if X0.shape != free.shape:
        raise ValueError(f"x0_phys has shape {X0.shape}, expected {free.shape} (samples, parameters)")
    X0 = np.where(free, X0, np.asarray(carried, float))
    fr = jnp.asarray(free)
    spans = [(a, min(a + max_chunk, n)) for a in range(0, n, max_chunk)] if max_chunk else [(0, n)]
    cut = lambda tree, a, b: jax.tree.map(lambda v: v[a:b], tree)

    def simulate(q, ev, m, e, car_i):                # one sample, one set of events -> Z
        if spread is None:
            return simulate_spectrum(q, ev, m, return_full_trajectory=False)[:, mz]
        sig = e[l_sp] if l_sp is not None else car_i[L]
        return simulate_spectrum_spread(q, ev, m, sig, spread)[:, mz]

    if not blocked:
        def model(loc, p, m0_i, free_i, car_i):      # one sample: its scaled slot values -> Z
            eff = jnp.where(free_i, (loc * spec.scales).astype(dt), car_i if spread is None else car_i[:L])
            q, m = _apply(p, m0_i, eff, free_i, spec)
            return simulate(q, events, m, eff, car_i)
        vjl = jax.jit(jax.vmap(jax.jacfwd(model)))  # (samples, points, slots): per sample, its own slots
    else:
        idx = np.array([[pos[l] + (b if is_block[l] else 0) for l in range(L)] for b in range(nb)])
        ev_b = jax.tree.map(lambda a: a.reshape((nb, n_per_block) + a.shape[1:]), events)

        def model(loc, p, m0_i, free_i, car_i):      # every block with its own per-block numbers
            cs = car_i if spread is None else car_i[:L]
            eff = jnp.where(free_i, (loc[idx] * spec.scales).astype(dt), cs)      # (blocks, slots)

            def one(e, ev):
                q, m = _apply(p, m0_i, e, free_i, spec)
                return simulate(q, ev, m, e, car_i)
            return jax.vmap(one)(eff, ev_b).reshape(-1)

        if block_jacobian == "full":
            # jacfwd over every local entry, a block slot's blocks included (under jit a jvp along tangent
            # directions captured as a constant came out wrong, jax 0.6.2 CPU, 2026-10-07; jacfwd was right)
            vjl = jax.jit(jax.vmap(jax.jacfwd(model)))
        else:
            # ONE tangent per slot: d moves every local entry of slot l together (E); block b's points depend
            # only on block b's entry, so W keeps of that derivative the rows of block b for entry pos+b.
            # jacfwd's own basis, no tangent captured as a constant; checked against "full" in the tests.
            E = np.zeros((L, n_loc))
            W = np.zeros((n_pts, L, n_loc))
            for l in range(L):
                for b in (range(nb) if is_block[l] else [0]):
                    E[l, pos[l] + b] = 1.0
                    rows = slice(b * n_per_block, (b + 1) * n_per_block) if is_block[l] else slice(None)
                    W[rows, l, pos[l] + b] = 1.0
            Ej, Wj = jnp.asarray(E, dt), jnp.asarray(W, dt)

            def jloc(loc, p, m0_i, free_i, car_i):    # (points, local entries)
                md = lambda d: model(loc + d @ Ej, p, m0_i, free_i, car_i)
                return jnp.einsum("nl,nlj->nj", jax.jacfwd(md)(jnp.zeros(L, dt)), Wj)
            vjl = jax.jit(jax.vmap(jloc))
    vz = jax.jit(jax.vmap(model))

    def per_sample(fn, loc):
        return jnp.concatenate([fn(loc[a:b], cut(params, a, b), m0[a:b], fr[a:b], carried_in[a:b])
                                for a, b in spans])

    Gj = jnp.asarray(G, dt)
    locs = lambda x: jnp.einsum("ijp,p->ij", Gj, x)

    def resid_all(x):                                  # every sample's residual, from the joint vector
        return (jax.vmap(model)(locs(x), params, m0, fr, carried_in) - z).reshape(-1)

    def jac_all(x):                                    # the joint Jacobian, assembled per sample
        return jnp.einsum("inj,ijp->inp", per_sample(vjl, locs(x)), Gj).reshape(-1, P)

    if _internals:
        x_start = jnp.asarray([X0[who[0], l] / scales[l] for l, who, _ in cols], dt)
        return dict(resid=resid_all, jac=jac_all, x0=x_start, cols=cols, model=model, locs=locs)

    coupled = any(s.scope in ("shared", "block") for s in spec.slots)
    nan_steps = 0
    if not coupled:
        # block diagonal: every sample alone, with its own damping -- fit's algorithm
        hold = None if free.all() else fr.astype(dt)
        lo = jnp.asarray(spec.lo_phys / scales, dt)
        hi = jnp.asarray(spec.hi_phys / scales, dt)

        def solve(p, m0_i, f_i, c_i, z_i, x0_i, h_i):
            r = lambda x: model(x, p, m0_i, f_i, c_i) - z_i
            return levenberg_marquardt(r, x0_i, lo, hi, n_iter=n_iter, lam0=lam0, use_fwd=True,
                                       return_trajectory=True, free=h_i)
        vsolve = jax.vmap(solve, in_axes=(0, 0, 0, 0, 0, 0, None if hold is None else 0))
        X0s = jnp.asarray(X0 / scales, dt)
        outs = [vsolve(cut(params, a, b), m0[a:b], fr[a:b], carried_in[a:b], z[a:b], X0s[a:b],
                       None if hold is None else hold[a:b]) for a, b in spans]
        loc = jnp.concatenate([o[0] for o in outs])
        cost = np.asarray(jnp.concatenate([o[2] for o in outs]), float).sum(axis=0)
    else:
        # coupled: one joint step over every free value, one damping
        col_slot = np.array([l for l, _, _ in cols], int)
        col_scale = scales[col_slot]
        lo = jnp.asarray(spec.lo_phys[col_slot] / col_scale, dt)
        hi = jnp.asarray(spec.hi_phys[col_slot] / col_scale, dt)
        eye = jnp.eye(P, dtype=dt)

        resid, jac = resid_all, jac_all

        @jax.jit
        def step(x, lam, J, r):
            c = r @ r
            g, H = J.T @ r, J.T @ J
            xn = jnp.clip(x + jnp.linalg.solve(H + lam * (jnp.diag(jnp.diag(H)) + 1e-9 * eye), -g), lo, hi)
            cn = jnp.sum(resid(xn) ** 2)
            better = cn < c
            return jnp.where(better, xn, x), jnp.where(better, lam * 0.5, lam * 3.0), jnp.where(better, cn, c)

        resid_j = jax.jit(resid)
        x = jnp.asarray([X0[who[0], l] / scales[l] for l, who, _ in cols], dt)
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

    loc_np = np.asarray(loc, float)
    X = np.where(free, loc_np[:, pos] * scales, np.asarray(carried, float))
    z_fit = np.asarray(per_sample(vz, loc), float)
    r_all = z_fit - np.asarray(z, float)

    # error bars and correlations at the solution, misfit-scaled
    sd = np.full(free.shape, np.nan)
    corr = np.full(free.shape + (L,), np.nan)
    Jl = np.asarray(per_sample(vjl, loc), float)        # (samples, points, local entries)
    blocks = {}
    if not coupled:
        for i in range(n):
            use = np.flatnonzero(free[i])
            if use.size:
                Ju = Jl[i][:, pos[use]]
                s2 = float(r_all[i] @ r_all[i]) / max(r_all.shape[1] - use.size, 1)
                sd[i, use], corr[i][np.ix_(use, use)] = _cov_to_sd_corr(np.linalg.pinv(Ju.T @ Ju) * s2, scales[use])
    else:
        J = np.einsum("inj,ijp->inp", Jl, G).reshape(-1, P)
        s2 = float(np.sum(r_all ** 2)) / max(r_all.size - P, 1)
        sd_col, corr_col = _cov_to_sd_corr(np.linalg.pinv(J.T @ J) * s2, col_scale)
        for i in range(n):
            ps = [p for p, (l, who, b) in enumerate(cols) if i in who and b < 0]
            ls = col_slot[ps]
            sd[i, ls] = sd_col[ps]
            corr[i][np.ix_(ls, ls)] = corr_col[np.ix_(ps, ps)]
        for l in np.flatnonzero(is_block):
            ps = [p for p, (ll, _, b) in enumerate(cols) if ll == l]
            xb = np.asarray(x, float)[ps] * scales[l]
            blocks[spec.slots[l].name] = dict(x=xb, sd=sd_col[ps], at_bounds=(
                np.isclose(xb, spec.slots[l].lo) | np.isclose(xb, spec.slots[l].hi)))
    X[:, is_block] = np.nan                             # per-block values live in `blocks`
    X0 = np.where(is_block[None, :], np.nan, X0)
    free = free & ~is_block[None, :]
    return FitResult(X, z_fit, np.asarray(z, float), spec, n_per_block, free=free, sd=sd, corr=corr,
                     x0=X0, cost=cost, nan_steps=nan_steps, blocks=blocks or None)
