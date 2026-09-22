"""
fitting.py  --  case-agnostic BMC fitting on top of the JAX simulator.

Design (one-directional imports: this file is the TOP layer):

    solver  ->  simulator (simulate_spectrum)  ->  voxel_simulator (simulate_voxels)
                                                          ^
                                                       fitting   (this file)

Nothing here touches the physics. A tiny `FitSpec` says WHICH parameters to fit
(no GAG/NOE hardcoding), and one residual core is shared by three backends:

  * fit_voxel        : single voxel, scipy Levenberg-Marquardt + exact jacrev.
                       (your old use case, done right -- LM not L-BFGS, residual
                       vector not a scaled MSE, no magic LOSS_SCALE.)
  * fit_phantom_lm   : MANY voxels, vmapped damped Gauss-Newton on the GPU.
                       Most accurate per voxel; exact Jacobian via jacrev.
  * fit_phantom_adam : MANY voxels, one optax/Adam loop over the whole phantom.
                       Most scalable; pure gradient steps, no per-voxel solves.

Per-voxel B0 and B1 are fit simply by listing them in the FitSpec (fields
"b0_inhom" and "rel_b1"). They are real per-voxel parameters, not baked into the
sequence.
"""
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .jax_solver.param_adapter import JAXParams
from .jax_solver.simulator import simulate_spectrum
from .jax_solver.voxel_simulator import simulate_voxels, tile_params, extract_z_spectra
from .jax_solver.utils import cast_params_to_dtype, cast_events_to_dtype


def _cast_problem(base, events, M0, dtype):
    """Downcast the whole problem (params, sequence, init state) to a dtype."""
    if dtype is None:
        return base, events, M0
    return cast_params_to_dtype(base, dtype), cast_events_to_dtype(events, dtype), M0.astype(dtype)


# ---------------------------------------------------------------------------
# FitSpec: declare which parameters are free (replaces update_jax_params_gag)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    name: str           # human label, e.g. "gag_k"
    field: str          # JAXParams field, e.g. "cest_k", "cest_f", "b0_inhom", "rel_b1"
    index: Optional[int] # pool index for array fields (cest_*); None for scalar fields
    scale: float        # rough magnitude, so the optimiser sees O(1) numbers
    lo: float           # lower bound (physical units)
    hi: float           # upper bound (physical units)
    x0: float           # initial guess (physical units)


@dataclass(frozen=True)
class FitSpec:
    slots: List[Slot]

    @property
    def n(self):                 return len(self.slots)
    @property
    def scales(self):            return jnp.array([s.scale for s in self.slots])
    @property
    def lo_phys(self):           return np.array([s.lo for s in self.slots])
    @property
    def hi_phys(self):           return np.array([s.hi for s in self.slots])
    @property
    def x0_phys(self):           return np.array([s.x0 for s in self.slots])
    # scaled versions (what the optimiser actually works in)
    def to_scaled(self, x_phys): return np.asarray(x_phys) / np.array([s.scale for s in self.slots])
    def to_phys(self, x_scaled): return np.asarray(x_scaled) * np.array([s.scale for s in self.slots])


# ---------------------------------------------------------------------------
# Write a flat (scaled) parameter vector into a JAXParams (traceable + vmappable)
# ---------------------------------------------------------------------------
def apply_x(base: JAXParams, x_scaled: jnp.ndarray, spec: FitSpec) -> JAXParams:
    """x_scaled -> physical -> written into the named slots. Pure JAX; safe under vmap."""
    dt = base.water_r1.dtype                      # match the params' dtype (fp32/fp64)
    x_phys = (x_scaled * spec.scales).astype(dt)
    updates = {}
    for i, s in enumerate(spec.slots):
        val = x_phys[i]
        if s.index is None:
            updates[s.field] = val
        else:
            cur = updates.get(s.field, getattr(base, s.field))
            updates[s.field] = cur.at[s.index].set(val)
    return base._replace(**updates)


# ---------------------------------------------------------------------------
# Residual core (shared by all backends)
# ---------------------------------------------------------------------------
def make_single_residual(base, events, M0, water_z, spec, z_meas, skip_first=True):
    """Returns r(x_scaled) -> residual vector for ONE voxel (z_sim - z_meas)."""
    z_t = jnp.asarray(z_meas[1:] if skip_first else z_meas)

    def resid(x_scaled):
        params = apply_x(base, x_scaled, spec)
        z = simulate_spectrum(params, events, M0, return_full_trajectory=False)[:, water_z]
        zc = z[1:] if skip_first else z
        return zc - z_t
    return resid


def make_phantom_residual(base, events, M0, water_z, spec, z_meas_batch, skip_first=True):
    """Returns R(X_scaled) -> (N_voxels, n_resid) residuals for the whole phantom.
    base is ONE shared JAXParams (fixed fields); X_scaled is (N_voxels, n_params)."""
    Z_t = jnp.asarray(z_meas_batch[:, 1:] if skip_first else z_meas_batch)
    n_vox = z_meas_batch.shape[0]
    base_b = tile_params(base, n_vox)                       # broadcast fixed fields

    def resid(X_scaled):
        params_b = jax.vmap(apply_x, in_axes=(0, 0, None))(base_b, X_scaled, spec)
        Z = simulate_voxels(params_b, events, M0, return_full_trajectory=False)
        Zc = extract_z_spectra(Z, water_z)                  # (N, n_offsets)
        Zc = Zc[:, 1:] if skip_first else Zc
        return Zc - Z_t                                     # (N, n_resid)
    return resid


# ---------------------------------------------------------------------------
# Backend 1: single-voxel Levenberg-Marquardt (scipy + exact Jacobian)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Backend 2: whole-phantom vmapped damped Gauss-Newton (most accurate)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=["resid_fn", "n_iter", "use_fwd", "return_trajectory"])
def _lm_one(resid_fn, x0, lo, hi, n_iter=40, lam0=1e-2, use_fwd=False, return_trajectory=False):
    """Damped Gauss-Newton (LM) for ONE voxel, fixed iterations with accept/reject.
    Bounds enforced by clipping the step. Pure JAX -> vmappable.

    use_fwd selects forward-mode (jacfwd) instead of reverse-mode (jacrev) for the Jacobian.
    They give the identical Jacobian; forward mode is far cheaper in MEMORY when the residual
    is tall (few params, many offsets) because it doesn't store the spectrum's forward
    trajectory -- essential for long multi-B1/Tsat schedules. Default stays jacrev.

    return_trajectory=True also returns (X_traj (n_iter, P) scaled, C_traj (n_iter,) cost) -- the
    per-iteration parameter + cost path. Used by benchmark_lm_convergence to check that the fixed
    n_iter is enough (the LM has no gtol/ftol). The scan already carried the cost as its stacked
    output; this just also stacks x, so it is nearly free (n_iter x P extra)."""
    jacfn = (jax.jacfwd if use_fwd else jax.jacrev)(resid_fn)
    P = x0.shape[0]
    eye = jnp.eye(P, dtype=x0.dtype)             # match x0 precision (else x64 forces float64 here)

    def body(state, _):
        x, lam = state
        r = resid_fn(x)
        c = jnp.sum(r ** 2)
        J = jacfn(x)
        g = J.T @ r
        H = J.T @ J
        A = H + lam * (jnp.diag(jnp.diag(H)) + 1e-9 * eye)
        delta = jnp.linalg.solve(A, -g)
        x_new = jnp.clip(x + delta, lo, hi)
        c_new = jnp.sum(resid_fn(x_new) ** 2)
        better = c_new < c
        x_out = jnp.where(better, x_new, x)
        lam_out = jnp.where(better, lam * 0.5, lam * 3.0)
        return (x_out, lam_out), (x_out, c_out_select(better, c_new, c))

    (xf, _), (X_traj, C_traj) = jax.lax.scan(body, (x0, jnp.asarray(lam0, x0.dtype)),
                                             None, length=n_iter)
    if return_trajectory:
        return xf, X_traj, C_traj
    return xf


def c_out_select(better, a, b):
    return jnp.where(better, a, b)


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


# ---------------------------------------------------------------------------
# Backend 2b: MULTI-GPU whole-phantom Gauss-Newton (voxels sharded across devices)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Backend 3: whole-phantom Adam (most scalable)
# ---------------------------------------------------------------------------
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
