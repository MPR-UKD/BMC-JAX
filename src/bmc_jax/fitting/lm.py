"""lm.py -- bounded Levenberg-Marquardt in JAX (formerly optimize.py, `lm_minimize`). Knows nothing about Bloch-McConnell.

THIS MODULE MUST NOT IMPORT THE SOLVER. It takes a residual function and bounds and minimises;
that is the whole contract. Keeping it that way is what makes `levenberg_marquardt` usable on its own,
and it is checked by tests/test_structure.py rather than left to good intentions.

Damped Gauss-Newton with a fixed iteration count and accept/reject, jitted and vmappable. Fixed
iterations rather than a convergence test because the whole point is to run thousands of these
in parallel under vmap, where an early exit for one element would stall the batch.

`use_fwd` picks forward-mode for the Jacobian. For a tall residual -- more offsets than free
parameters, which is always true in CEST -- forward mode costs far less memory and is usually
faster. The result is identical either way.
"""
from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=["resid_fn", "n_iter", "use_fwd", "return_trajectory"])
def levenberg_marquardt(resid_fn, x0, lo, hi, n_iter=40, lam0=1e-2, use_fwd=False, return_trajectory=False,
                        free=None):
    """Damped Gauss-Newton (LM) for ONE voxel, fixed iterations with accept/reject.
    Bounds enforced by clipping the step. Pure JAX -> vmappable.

    use_fwd selects forward-mode (jacfwd) instead of reverse-mode (jacrev) for the Jacobian.
    They give the identical Jacobian; forward mode is far cheaper in MEMORY when the residual
    is tall (few params, many offsets) because it doesn't store the spectrum's forward
    trajectory -- essential for long multi-B1/Tsat schedules. Default stays jacrev.

    return_trajectory=True also returns (X_traj (n_iter, P) scaled, C_traj (n_iter,) cost) -- the
    per-iteration parameter + cost path. Used by benchmark_lm_convergence to check that the fixed
    n_iter is enough (the LM has no gtol/ftol). The scan already carried the cost as its stacked
    output; this just also stacks x, so it is nearly free (n_iter x P extra).

    free (optional, shape (P,), 1.0 / 0.0): a 0 entry is held at x0 -- its step is exactly zero whatever
    the damping (fit_joint, for a parameter that is free in some samples only). None = all free, and
    the arithmetic is exactly what it was without the argument."""
    jacfn = (jax.jacfwd if use_fwd else jax.jacrev)(resid_fn)
    P = x0.shape[0]
    eye = jnp.eye(P, dtype=x0.dtype)             # match x0 precision (else x64 forces float64 here)

    def body(state, _):
        x, lam = state
        r = resid_fn(x)
        c = jnp.sum(r ** 2)
        J = jacfn(x)
        if free is not None:                     # a held entry: no Jacobian column (so no coupling) ...
            J = J * free
        g = J.T @ r
        H = J.T @ J
        A = H + lam * (jnp.diag(jnp.diag(H)) + 1e-9 * eye)
        if free is not None:                     # ... and a unit diagonal: its step is exactly zero
            A = A + jnp.diag(1.0 - free)
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

