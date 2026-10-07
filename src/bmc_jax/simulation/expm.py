"""One time step of the Bloch-McConnell system, with two interchangeable matrix exponentials.

`propagate(M, dt, A, C)` advances the magnetisation by dt under dM/dt = A M + C. In homogeneous
coordinates (matrix.py) that is one matrix exponential of the augmented (N+1) x (N+1) matrix and
never an inversion.

TWO SOLVERS, ONE SWITCH: `bmc_jax.settings.use_expm(...)`
  "library"  jax.scipy.linalg.expm. Pade 3/5/7 with scaling and squaring; the matrix products and the
             solve are done by the linear-algebra libraries (cuBLAS / cuSOLVER on a GPU). The default,
             and what every published number used.
  "plain"    the same float32 algorithm -- same Pade orders, constants and squaring rule -- with every
             matrix product and the solve written out as elementwise operations, so no library call is
             made. Why it exists: on an RTX 3070 (jax 0.6.2 and 0.7.2) the library calls inside expm
             returned NaN or wrong forward-mode slopes in a vmapped Jacobian for some batch shapes (several
             samples x hundreds of offsets), with a count that changed between identical calls, while the
             residual itself was bit-identical. "plain" showed none of that. float32 only.
  The choice is read when a function is traced (compiled); use_expm clears JAX's compilation cache so
  the next call picks it up.
"""
import jax
import jax.numpy as jnp

from ..settings import expm_backend


def expm_library(A):
    """jax.scipy.linalg.expm: the matrix exponential through the linear-algebra libraries."""
    return jax.scipy.linalg.expm(A)


def _dot(X, Y):
    """X @ Y as multiply + sum: no library (cuBLAS gemm) call."""
    return jnp.sum(X[..., :, :, None] * Y[..., None, :, :], axis=-2)


def _solve_gj(Q, P):
    """Q^-1 P by Gauss-Jordan with partial pivoting: no library (cuSOLVER) call."""
    n = Q.shape[-1]
    M = jnp.concatenate([Q, P], axis=-1)
    rows = jnp.arange(n)
    for c in range(n):
        piv = c + jnp.argmax(jnp.abs(M[c:, c]))
        rc, rp = M[c], M[piv]
        M = M.at[c].set(rp).at[piv].set(rc)
        M = M.at[c].set(M[c] / M[c, c])
        M = M - jnp.where((rows != c)[:, None], M[:, c:c + 1] * M[c][None, :], 0.0)
    return M[:, n:]


def expm_plain(A, max_squarings=16):
    """jax.scipy.linalg.expm's float32 algorithm (Pade 3/5/7, scaling and squaring, same constants),
    with every matrix product and the solve written out, so no linear-algebra library is called."""
    if A.dtype != jnp.float32:
        raise ValueError('use_expm("plain") is float32 only (x64 is on)')
    I = jnp.eye(A.shape[0], dtype=A.dtype)

    def pade3(A):
        b = (120., 60., 12., 1.)
        A2 = _dot(A, A)
        return _dot(A, b[3] * A2 + b[1] * I), b[2] * A2 + b[0] * I

    def pade5(A):
        b = (30240., 15120., 3360., 420., 30., 1.)
        A2 = _dot(A, A); A4 = _dot(A2, A2)
        return _dot(A, b[5] * A4 + b[3] * A2 + b[1] * I), b[4] * A4 + b[2] * A2 + b[0] * I

    def pade7(A):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        A2 = _dot(A, A); A4 = _dot(A2, A2); A6 = _dot(A4, A2)
        return (_dot(A, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I),
                b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I)

    a_l1 = jnp.max(jnp.sum(jnp.abs(A), axis=0))                 # = jnp.linalg.norm(A, 1)
    n_sq = jnp.maximum(0, jnp.floor(jnp.log2(a_l1 / 3.925724783138660)))
    As = A / 2 ** n_sq.astype(A.dtype)
    idx = jnp.digitize(a_l1, jnp.array([4.258730016922831e-001, 1.880152677804762e+000], a_l1.dtype))
    U, V = jax.lax.switch(idx, [pade3, pade5, pade7], As)
    R = _solve_gj(V - U, U + V)

    def sq(c, i):
        return jax.lax.cond(i < n_sq, lambda x: _dot(x, x), lambda x: x, c), None
    R, _ = jax.lax.scan(sq, R, jnp.arange(max_squarings, dtype=n_sq.dtype))
    return jnp.where(n_sq > max_squarings, jnp.nan, R)


def _propagate_plain(M, dt, A, C):
    """The time step with expm_plain and a written-out matrix-vector product."""
    size = M.shape[0]
    aug = jnp.zeros((size + 1, size + 1), dtype=A.dtype).at[:size, :size].set(A).at[:size, size].set(C)
    E = expm_plain(aug * dt)
    return jnp.sum(E * jnp.append(M, 1.0)[None, :], axis=1)[:size]


def _propagate_library(M: jnp.ndarray, dt: float, A: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """
    Solves the linear ODE dM/dt = A*M + C for a single time step dt.
    """
    size = M.shape[0]
    
    # Construct augmented matrix (N+1, N+1)
    aug_A = jnp.zeros((size + 1, size + 1), dtype=A.dtype)
    aug_A = aug_A.at[:size, :size].set(A)
    aug_A = aug_A.at[:size, size].set(C)
    
    # Compute matrix exponential
    exp_aug = expm_library(aug_A * dt)
    
    # Propagate state
    M_aug = jnp.append(M, 1.0)
    M_new_aug = exp_aug @ M_aug
    
    return M_new_aug[:size]


def propagate(M: jnp.ndarray, dt: float, A: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """Solve dM/dt = A*M + C over one time step dt, with the matrix exponential `use_expm` selected."""
    if expm_backend() == "plain":
        return _propagate_plain(M, dt, A, C)
    return _propagate_library(M, dt, A, C)
