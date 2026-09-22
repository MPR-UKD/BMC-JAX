"""batching.py -- the same fit, over one spectrum, many voxels, or many GPUs.

These four entry points differ in exactly two ways: which optimiser runs, and how the work is
spread. `fit_voxel` is one spectrum on the CPU via scipy; `fit_phantom_lm` vmaps the JAX LM over
every voxel; `fit_phantom_lm_sharded` splits that batch across the local GPUs; `fit_phantom_adam`
replaces LM with one Adam loop over the whole phantom.

That is a two-by-two grid written out as four functions, and a later step folds it into
`fit(..., optimizer=, batch=)`. The four names stay exported regardless.

`max_chunk` is the knob that matters in practice: vmapping over every voxel at once allocates
for every voxel at once, and a large phantom will exhaust the card. Chunking bounds peak memory
independently of how many voxels there are.
"""
import numpy as np
import jax
import jax.numpy as jnp

from ..jax_solver.simulator import simulate_spectrum
from ..jax_solver.voxel_simulator import simulate_voxels, tile_params, extract_z_spectra
from ..jax_solver.utils import cast_params_to_dtype, cast_events_to_dtype
from .spec import FitSpec, Slot, apply_x
from .residual import _cast_problem, make_single_residual, make_phantom_residual
from .optimize import _lm_one, c_out_select, lm_minimize
from .experiment import Experiment
from .result import FitResult

def fit_voxel(base, events, M0, water_z, spec, z_meas, x0_phys=None, skip_first=True, dtype=None):
    from scipy.optimize import least_squares
    base, events, M0 = _cast_problem(base, events, M0, dtype)
    if dtype is not None:
        z_meas = jnp.asarray(z_meas).astype(dtype)
    resid = make_single_residual(base, events, M0, water_z, spec, z_meas, skip_first)
    rj = jax.jit(resid)
    jj = jax.jit(jax.jacrev(resid))
    castx = (lambda x: jnp.asarray(x, dtype)) if dtype is not None else (lambda x: jnp.asarray(x))
    x0 = spec.to_scaled(spec.x0_phys if x0_phys is None else x0_phys)
    res = least_squares(
        lambda x: np.asarray(rj(castx(x))),
        x0,
        jac=lambda x: np.asarray(jj(castx(x))),
        bounds=(spec.lo_phys / np.array([s.scale for s in spec.slots]),
                spec.hi_phys / np.array([s.scale for s in spec.slots])),
        method="trf",
    )
    return spec.to_phys(res.x), res


def fit_phantom_lm(base, events, M0_batch, water_z, spec, z_meas_batch,
                   X0_phys=None, n_iter=40, skip_first=True, base_batched=False,
                   use_fwd=None, max_chunk=None, dtype=None):
    """Fit every voxel with damped Gauss-Newton, all voxels in parallel (vmap).

    base_batched=False -> `base` is ONE tissue, broadcast to every voxel (synthetic /
    homogeneous case). base_batched=True -> `base` already carries a leading (N_voxels,)
    axis on every field, so per-voxel FIXED maps (measured T1/T2 -> water_r1/water_r2,
    WASSR B0 -> b0_inhom, WASABI -> rel_b1) are honoured instead of being overwritten by
    a single shared value. Fields listed in `spec` are still fit per voxel on top of it.

    use_fwd -> forward-mode Jacobian (see _lm_one; identical result). Default None AUTO-selects
    it whenever the residual is tall (offsets >= params), which is always true for CEST -> far
    less memory and usually faster. Pass True/False to force.
    max_chunk -> if set, fit at most this many voxels per vmap call (looping + concatenating),
    so peak GPU memory stays bounded regardless of N_voxels. None = one shot (old behaviour).
    dtype -> if set (jnp.float32/float64), cast the whole problem to that precision and fit at it
    (used by the accuracy benchmarks to compare fp32 vs fp64 in one x64-enabled run)."""
    n_vox = z_meas_batch.shape[0]
    # per-voxel residual: close over each voxel's measured spectrum
    base_b = base if base_batched else tile_params(base, n_vox)
    if dtype is not None:                        # run the whole fit at this precision
        base_b = cast_params_to_dtype(base_b, dtype)
        events = cast_events_to_dtype(events, dtype)
        M0_batch = M0_batch.astype(dtype)
        z_meas_batch = jnp.asarray(z_meas_batch).astype(dtype)
    Z_t = jnp.asarray(z_meas_batch[:, 1:] if skip_first else z_meas_batch)
    if use_fwd is None:
        use_fwd = Z_t.shape[1] >= spec.n         # tall residual -> forward mode is cheaper

    # M0 IS PER VOXEL AND MUST BE MAPPED, NOT INDEXED. This used to close over M0_batch[0], so
    # every voxel was simulated with voxel 0's starting magnetisation while base, Z and X0 were
    # correctly vmapped. Callers that build M0b with jnp.broadcast_to (all the benchmarks) are
    # unaffected -- every element is identical, so [0] was right by construction and this change
    # is a provable no-op for them. Callers that build a genuinely per-voxel M0 with jnp.stack
    # were silently getting row 0: found 2026-08-10 in the 6-vial phantom fit, where the vials
    # differ in T1 and therefore in post-readout M_start. Measured cost there: GAG was fitted at
    # M_start 0.5801 instead of its own 0.6246 (-7.1%), which after the 4 s recovery delay is a
    # 1.0% error in the magnetisation entering saturation. Small, but it fell on the one vial
    # whose starting-state deficit was under investigation.
    M0_b = jnp.asarray(M0_batch)
    if M0_b.ndim == 1:                            # a single M0 handed in unbatched
        M0_b = jnp.broadcast_to(M0_b, (n_vox,) + M0_b.shape)
    if M0_b.shape[0] != n_vox:
        raise ValueError(f"M0_batch has {M0_b.shape[0]} rows for {n_vox} voxels")
    if dtype is not None:
        M0_b = M0_b.astype(dtype)

    def resid_for(base_v, z_t_v, m0_v):
        def r(x_scaled):
            params = apply_x(base_v, x_scaled, spec)
            z = simulate_spectrum(params, events, m0_v, return_full_trajectory=False)[:, water_z]
            zc = z[1:] if skip_first else z
            return zc - z_t_v
        return r

    scales = np.array([s.scale for s in spec.slots])
    lo = jnp.asarray(spec.lo_phys / scales)
    hi = jnp.asarray(spec.hi_phys / scales)
    if X0_phys is None:
        X0 = jnp.broadcast_to(jnp.asarray(spec.to_scaled(spec.x0_phys)), (n_vox, spec.n))
    else:
        X0 = jnp.asarray(spec.to_scaled(X0_phys))
    if dtype is not None:
        lo, hi, X0 = lo.astype(dtype), hi.astype(dtype), X0.astype(dtype)

    def solve_one(base_v, z_t_v, x0, m0_v):
        return _lm_one(resid_for(base_v, z_t_v, m0_v), x0, lo, hi, n_iter=n_iter, use_fwd=use_fwd)
    solve_batch = jax.vmap(solve_one, in_axes=(0, 0, 0, 0))

    if not max_chunk or n_vox <= max_chunk:
        Xf = solve_batch(base_b, Z_t, X0, M0_b)
    else:                                        # fit in voxel chunks -> bounded peak memory
        parts = [solve_batch(jax.tree.map(lambda a: a[s:s + max_chunk], base_b),
                             Z_t[s:s + max_chunk], X0[s:s + max_chunk], M0_b[s:s + max_chunk])
                 for s in range(0, n_vox, max_chunk)]
        Xf = jnp.concatenate(parts, axis=0)
    return np.asarray(Xf) * scales               # physical (N, P)


def fit_phantom_lm_sharded(base, events, M0_batch, water_z, spec, z_meas_batch,
                           n_iter=40, skip_first=True, n_devices=None, base_batched=False,
                           use_fwd=None, per_device_chunk=256):
    """Same damped Gauss-Newton fit as fit_phantom_lm, but the voxel axis is SPLIT across
    all local GPUs via jax.jit + NamedSharding (see jax_solver.sharded.sharded_vmap).

    This replaces the earlier jax.pmap implementation. pmap pinned each device's batch at
    total/n_dev and OOM'd when that was too big (and its 1-GPU fallback fit every voxel in one
    shot -> the 23.8 GiB failure in benchmark_sharding). sharded_vmap instead loops the batch in
    super-chunks so each GPU holds at most `per_device_chunk` voxels at once -- peak memory is set
    by that knob, not by N_voxels. Voxels don't interact, so the sharded result is bit-for-bit the
    single-device fit (validate with benchmark_sharding.py --validate); padding rows are dropped.

    per_device_chunk defaults small (256) because real fits are the heavy PULSED 607-offset kind;
    raise it for light (block-pulse) problems to reduce loop overhead. n_devices=None uses every
    visible GPU; with one device this is just fit_phantom_lm chunked to `per_device_chunk`.
    use_fwd (default None) auto-selects the forward-mode Jacobian for tall residuals, as elsewhere."""
    from jax_solver.sharded import sharded_vmap, pick_devices   # core primitive (mirrored to all images)

    if len(pick_devices(n_devices)) <= 1:            # single GPU -> the chunked single-device path
        return fit_phantom_lm(base, events, M0_batch, water_z, spec, z_meas_batch,
                              n_iter=n_iter, skip_first=skip_first, base_batched=base_batched,
                              use_fwd=use_fwd, max_chunk=per_device_chunk)

    n_vox = z_meas_batch.shape[0]
    scales = np.array([s.scale for s in spec.slots])
    base_b = base if base_batched else tile_params(base, n_vox)
    Z_t = jnp.asarray(z_meas_batch[:, 1:] if skip_first else z_meas_batch)
    # Same per-voxel M0 fix as in fit_phantom_lm above -- see the note there.
    M0_b = jnp.asarray(M0_batch)
    if M0_b.ndim == 1:
        M0_b = jnp.broadcast_to(M0_b, (n_vox,) + M0_b.shape)
    if M0_b.shape[0] != n_vox:
        raise ValueError(f"M0_batch has {M0_b.shape[0]} rows for {n_vox} voxels")
    if use_fwd is None:
        use_fwd = Z_t.shape[1] >= spec.n             # tall residual -> forward mode is cheaper
    lo = jnp.asarray(spec.lo_phys / scales)
    hi = jnp.asarray(spec.hi_phys / scales)
    X0 = jnp.broadcast_to(jnp.asarray(spec.to_scaled(spec.x0_phys)), (n_vox, spec.n))

    # identical per-voxel solve to fit_phantom_lm; sharded_vmap partitions the batch axis for us.
    def solve_one(base_v, z_t_v, x0, m0_v):
        def r(x_scaled):
            params = apply_x(base_v, x_scaled, spec)
            zz = simulate_spectrum(params, events, m0_v, return_full_trajectory=False)[:, water_z]
            zc = zz[1:] if skip_first else zz
            return zc - z_t_v
        return _lm_one(r, x0, lo, hi, n_iter=n_iter, use_fwd=use_fwd)
    solve_batch = jax.vmap(solve_one, in_axes=(0, 0, 0, 0))

    Xf = sharded_vmap(solve_batch, base_b, Z_t, X0, M0_b,
                      n_devices=n_devices, per_device_chunk=per_device_chunk)   # (n_vox, P) scaled
    return Xf * scales                               # physical (n_vox, P)


def fit_phantom_adam(base, events, M0_batch, water_z, spec, z_meas_batch,
                     X0_phys=None, n_steps=400, lr=2e-2, skip_first=True):
    import optax
    resid = make_phantom_residual(base, events, M0_batch, water_z, spec,
                                  z_meas_batch, skip_first)
    n_vox = z_meas_batch.shape[0]
    lo = jnp.asarray(spec.lo_phys / np.array([s.scale for s in spec.slots]))
    hi = jnp.asarray(spec.hi_phys / np.array([s.scale for s in spec.slots]))

    def loss(X):
        return jnp.sum(resid(X) ** 2)

    if X0_phys is None:
        X = jnp.broadcast_to(jnp.asarray(spec.to_scaled(spec.x0_phys)), (n_vox, spec.n))
    else:
        X = jnp.asarray(spec.to_scaled(X0_phys))
    X = jnp.array(X)

    opt = optax.adam(lr)
    state = opt.init(X)
    loss_grad = jax.jit(jax.value_and_grad(loss))

    @jax.jit
    def step(X, state):
        l, g = loss_grad(X)
        upd, state = opt.update(g, state)
        X = optax.apply_updates(X, upd)
        X = jnp.clip(X, lo, hi)        # projected step keeps params in bounds
        return X, state, l

    losses = []
    for _ in range(n_steps):
        X, state, l = step(X, state)
        losses.append(float(l))
    return np.asarray(X) * np.array([s.scale for s in spec.slots]), np.asarray(losses)


def fit(experiment, spec, z_meas, n_iter=30, x0_phys=None, n_per_block=None, **kwargs):
    """Fit `spec`'s parameters of `experiment` to `z_meas`. The entry point to prefer.

    Parameters
    ----------
    problem
        A :class:`~bmc_jax.fitting.experiment.Experiment`. Use `Experiment.stack` to fit several
        experiments at once, which is what separating an exchange rate from a pool size requires.
    spec
        Which parameters are free, with their bounds and starting values.
    z_meas
        Measured Z-spectra, shape (n_samples, n_offsets). One sample is accepted as (n_offsets,).
    n_iter
        Levenberg-Marquardt iterations.
    x0_phys
        Starting values in physical units, shape (n_samples, n_parameters). Defaults to the
        `x0` of each slot. A good starting value is legitimate; what it must not do is decide
        the answer, which is worth checking by refitting from a deliberately poor one.
    n_per_block
        Offsets per experiment, so the result can report the residual per rung of a ladder.
    **kwargs
        Passed through to :func:`fit_phantom_lm` (`max_chunk`, `dtype`, `use_fwd`, ...).

    Returns
    -------
    FitResult

    Warnings
    --------
    In float32, `max_chunk` is not numerically inert. The voxels are independent and chunking is
    the same computation in pieces, but changing the batch size changes the kernel XLA compiles,
    float32 reductions are not associative, and a k/f fit is close enough to singular that a
    ~1e-7 difference in the Jacobian moves the result by ~1 % in k. Measured on a 64-voxel
    problem: 1.1e-2 relative between one shot and max_chunk=32 in float32, 1.8e-7 in float64.

    So float32 fitting of an exchange rate is reproducible to about a percent, and that floor is
    set by conditioning rather than by the data. If a conclusion turns on a smaller difference,
    refit with :func:`bmc_jax.precision.configure_x64` enabled and check it survives.

    Notes
    -----
    This wraps `fit_phantom_lm`, which remains available and unchanged. The difference is that
    the state-vector index, the batching of the parameters and the shape handling come from the
    `Experiment` rather than from the caller.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> res = fit(Experiment.stack(ladder), spec, Z, n_per_block=len(ppm))
    >>> print(res.summary())
    """
    z = jnp.atleast_2d(jnp.asarray(z_meas))
    n = z.shape[0]

    base = experiment.params
    batched = jnp.asarray(base.water_r1).ndim > 0
    if not batched:
        base = tile_params(base, n)

    m0 = jnp.asarray(experiment.m0)
    if m0.ndim == 1:
        m0 = jnp.broadcast_to(m0, (n,) + m0.shape)

    X = np.asarray(fit_phantom_lm(base, experiment.events, m0, experiment.mz_loc, spec, z,
                                  X0_phys=x0_phys, n_iter=n_iter, skip_first=False,
                                  base_batched=True, **kwargs))

    xs = spec.to_scaled(X)
    z_fit = np.asarray(jax.vmap(lambda b, x, s: simulate_spectrum(
        apply_x(b, jnp.asarray(x), spec), experiment.events, s,
        return_full_trajectory=False)[:, experiment.mz_loc])(base, jnp.asarray(xs), m0))
    return FitResult(X, z_fit, np.asarray(z), spec, n_per_block)
