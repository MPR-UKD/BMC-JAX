"""
JAX-based Bloch-McConnell Solver.

This module provides the low-level mathematical functions to build the BMC matrix
and solve the differential equations using matrix exponentials.
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from .param_adapter import JAXParams

# Pre-compute SuperLorentzian constants to aid JIT constant folding
_SL_U = jnp.linspace(0.0, 1.0, 101)
_SL_DU = 0.01
# |3*u^2 - 1|
_SL_TERM1 = jnp.abs(3 * _SL_U**2 - 1)
# Avoid division by zero near magic angle (u = 1/sqrt(3))
_SL_TERM1 = jnp.maximum(_SL_TERM1, 1e-6)


def _interpolate_sl(delta_omega, t2):
    arg = (delta_omega * t2) / _SL_TERM1
    term2 = jnp.exp(-2 * arg**2)
    integrand = jnp.sqrt(2 / jnp.pi) * t2 / _SL_TERM1 * term2
    return jnp.sum(integrand) * jnp.pi * _SL_DU

def _interpolate_chs(delta_omega, t2, w0):
    px = jnp.array([-300 - w0, -100 - w0, 100 + w0, 300 + w0])
    
    # Calculate SL for each point
    py0 = _interpolate_sl(px[0], t2)
    py1 = _interpolate_sl(px[1], t2)
    py2 = _interpolate_sl(px[2], t2)
    py3 = _interpolate_sl(px[3], t2)
    
    tan_weight = 30.0
    d0y = tan_weight * (py1 - py0)
    d1y = tan_weight * (py3 - py2)
    
    c_step = jnp.abs((delta_omega - px[1] + 1) / (px[2] - px[1] + 1))
    
    h0 = 2 * (c_step**3) - 3 * (c_step**2) + 1
    h1 = -2 * (c_step**3) + 3 * (c_step**2)
    h2 = (c_step**3) - 2 * (c_step**2) + c_step
    h3 = (c_step**3) - (c_step**2)
    
    return h0 * py1 + h1 * py2 + h2 * d0y + h3 * d1y

def get_super_lorentzian_lineshape(delta_omega: float, t2: float, w0: float) -> float:
    return jax.lax.cond(
        jnp.abs(delta_omega) >= w0,
        lambda _: _interpolate_sl(delta_omega, t2),
        lambda _: _interpolate_chs(delta_omega, t2, w0),
        operand=None
    )

def get_lorentzian_lineshape(delta_omega: float, t2: float) -> float:
    """
    Calculates the Lorentzian lineshape value.
    
    Args:
        delta_omega: Frequency offset in rad/s.
        t2: T2 relaxation time in s.
        
    Returns:
        Lineshape value (s).
    """
    return t2 / (1 + (delta_omega * t2)**2)

def build_bmc_matrix_and_vector(params: JAXParams, rf_amp: float, rf_phase: float, rf_freq: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constructs the Bloch-McConnell matrix (A) and constant vector (C) for a given time step.
    
    The system is defined as dM/dt = A * M + C.
    
    Args:
        params: JAXParams object containing physical parameters.
        rf_amp: RF amplitude in Hz (gamma * B1 / 2pi).
        rf_phase: RF phase in radians.
        rf_freq: RF frequency offset in Hz.
        
    Returns:
        Tuple (A, C) where A is the (N, N) system matrix and C is the (N,) constant vector.
        N = 3 * (1 + num_cest_pools) + 1 (MT).
    """
    # Unpack parameters
    r1_w = params.water_r1
    r2_w = params.water_r2
    
    k_cw = params.cest_k
    f_c = params.cest_f
    r1_c = params.cest_r1
    r2_c = params.cest_r2
    dw_c = params.cest_dw # ppm
    
    # MT Parameters
    k_mtw = params.mt_k
    f_mt = params.mt_f
    r1_mt = params.mt_r1
    r2_mt = params.mt_r2
    dw_mt = params.mt_dw # ppm
    
    # Derived exchange rates
    k_wc = k_cw * f_c
    k_wmt = k_mtw * f_mt
    
    # RF parameters
    w1 = 2 * jnp.pi * rf_amp * params.rel_b1
    w1x = w1 * jnp.cos(rf_phase)
    w1y = w1 * jnp.sin(rf_phase)
    
    # Frequency parameters
    # params.gamma is in rad/uT -> convert to rad/T
    gamma_rad = params.gamma * 1e6 
    larmor_freq = gamma_rad * params.b0 # rad/s
    
    # B0 inhomogeneity offset in rad/s
    dw0 = larmor_freq * params.b0_inhom * 1e-6
    
    # Offsets in rad/s
    dw_w = 2 * jnp.pi * rf_freq # rad/s (offset of RF from water)
    
    """
    # === Legacy BMCTool Behavior (Incorrect Physics) ===
    # To account for B0 inhomogeneity (dw0), the effective RF frequency is shifted:
    # effective_dw_w = dw_w - dw0
    
    # CEST offsets
    # dw_c is in ppm. 
    # === b0 inhomogenities problem ===
    # offset = dw_c * 1e-6 * larmor_freq
    delta_omega_c = dw_c * 1e-6 * larmor_freq
    off_c = delta_omega_c - (dw_w + dw0)
    
    # Water offset
    # Calculate offset in the selected operational frame relative to explicit Larmor markers.
    # Corresponds directly with prior architectural logic configurations.
    off_w = dw_w - dw0
    
    # MT Lineshape
    # dw_mt is in ppm.
    delta_omega_mt = dw_mt * 1e-6 * larmor_freq
    off_mt = (dw_w + dw0) - delta_omega_mt # rad/s
    """
    
    # === CORRECTED PHYSICS: Uniform b0_inhom Shift ===
    # All pools should experience the exact same effective RF frequency offset.
    # The true offset experienced by the spins is (dw_w - dw0).
    effective_rf_freq = dw_w - dw0
    
    # CEST offsets: Distance from CEST resonance to effective RF
    delta_omega_c = dw_c * 1e-6 * larmor_freq
    off_c = delta_omega_c - effective_rf_freq
    
    # Water offset: just the effective RF directly
    off_w = effective_rf_freq
    
    # MT pool offset: Distance from MT resonance to effective RF
    delta_omega_mt = dw_mt * 1e-6 * larmor_freq
    off_mt = effective_rf_freq - delta_omega_mt
    
    # Calculate W_mt = pi * w1^2 * g(delta)
    t2_mt = 1.0 / r2_mt
    
    # Use SuperLorentzian lineshape
    # g_mt = get_super_lorentzian_lineshape(off_mt, t2_mt)
    
    # Select lineshape based on params.mt_lineshape
    # 0=None, 1=Lorentzian, 2=SuperLorentzian
    def get_none(off, t2):
        return jnp.array(0.0, dtype=off.dtype)
        
    def get_lor(off, t2):
        return get_lorentzian_lineshape(off, t2).astype(off.dtype)
        
    def get_sl(off, t2):
        return get_super_lorentzian_lineshape(off, t2, larmor_freq * 1e-6).astype(off.dtype)

    # Conditional computation routing. 
    # Bypasses complex SuperLorentzian approximations locally when MT pools are excluded.
    def calc_lineshape(dummy):
        # params.mt_lineshape is a 0-d array (scalar)
        ls_idx = params.mt_lineshape.astype(jnp.int32)
        return jax.lax.switch(
            ls_idx,
            [get_none, get_lor, get_sl],
            off_mt, t2_mt
        )

    g_mt = jax.lax.cond(
        f_mt > 1e-9,
        calc_lineshape,
        lambda _: jnp.array(0.0, dtype=off_mt.dtype),
        None
    )

    w_mt = w1**2 * g_mt
    
    # Enforce zero saturation rate for disabled MT pools to prevent numerical instability.
    w_mt = jnp.where(f_mt > 1e-9, w_mt, 0.0)
    
    # --- Build Matrix A ---
    n_cest = k_cw.shape[0]
    n_total = 1 + n_cest
    # Always include MT dimension to match fixed size layout
    size = 3 * n_total + 1
    
    # Ensure Matrix uses the input dtype (float32 or float64)
    A = jnp.zeros((size, size), dtype=r1_w.dtype)
    
    # Indices
    # Water: 0, n_total, 2*n_total
    ix_w, iy_w, iz_w = 0, n_total, 2 * n_total
    
    # CEST: 1..n_cest, ...
    ix_c = jnp.arange(1, n_total)
    iy_c = jnp.arange(n_total + 1, 2 * n_total)
    iz_c = jnp.arange(2 * n_total + 1, 3 * n_total)
    
    # MT: Last element
    iz_mt = 3 * n_total
    
    k_wc_sum = jnp.sum(k_wc)
    
    # --- Water Block ---
    # X
    A = A.at[ix_w, ix_w].set(-r2_w - k_wc_sum)
    A = A.at[ix_w, iy_w].set(off_w)
    A = A.at[ix_w, iz_w].set(-w1y)
    
    # Y
    A = A.at[iy_w, iy_w].set(-r2_w - k_wc_sum)
    A = A.at[iy_w, ix_w].set(-off_w)
    A = A.at[iy_w, iz_w].set(w1x)
    
    # Z (includes MT exchange)
    A = A.at[iz_w, iz_w].set(-r1_w - k_wc_sum - k_wmt)
    A = A.at[iz_w, iy_w].set(-w1x)
    A = A.at[iz_w, ix_w].set(w1y)
    
    # --- CEST Pools ---
    # X
    A = A.at[ix_c, ix_c].set(-r2_c - k_cw)
    A = A.at[ix_c, iy_c].set(-off_c)
    A = A.at[ix_c, iz_c].set(-w1y)
    
    # Y
    A = A.at[iy_c, iy_c].set(-r2_c - k_cw)
    A = A.at[iy_c, ix_c].set(off_c)
    A = A.at[iy_c, iz_c].set(w1x)
    
    # Z
    A = A.at[iz_c, iz_c].set(-r1_c - k_cw)
    A = A.at[iz_c, ix_c].set(w1y)
    A = A.at[iz_c, iy_c].set(-w1x)
    
    # --- Exchange Water <-> CEST ---
    # Water gains from CEST
    A = A.at[ix_w, ix_c].set(k_cw)
    A = A.at[iy_w, iy_c].set(k_cw)
    A = A.at[iz_w, iz_c].set(k_cw)
    
    # CEST gains from Water
    A = A.at[ix_c, ix_w].set(k_wc)
    A = A.at[iy_c, iy_w].set(k_wc)
    A = A.at[iz_c, iz_w].set(k_wc)
    
    # --- MT Pool (Z only) ---
    # dM_mt/dt = - (R1_mt + k_mtw + W_mt) * M_mt + k_wmt * M_wz
    A = A.at[iz_mt, iz_mt].set(-r1_mt - k_mtw - w_mt)
    A = A.at[iz_mt, iz_w].set(k_wmt)
    
    # Water Z gains from MT
    A = A.at[iz_w, iz_mt].set(k_mtw)
    
    # --- Vector C ---
    C = jnp.zeros(size, dtype=r1_w.dtype)
    C = C.at[iz_w].set(r1_w * 1.0)
    C = C.at[iz_c].set(r1_c * f_c)
    C = C.at[iz_mt].set(r1_mt * f_mt)
    
    return A, C

def solve_step(M: jnp.ndarray, dt: float, A: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """
    Solves the linear ODE dM/dt = A*M + C for a single time step dt.
    """
    size = M.shape[0]
    
    # Construct augmented matrix (N+1, N+1)
    aug_A = jnp.zeros((size + 1, size + 1), dtype=A.dtype)
    aug_A = aug_A.at[:size, :size].set(A)
    aug_A = aug_A.at[:size, size].set(C)
    
    # Compute matrix exponential
    exp_aug = jax.scipy.linalg.expm(aug_A * dt)
    
    # Propagate state
    M_aug = jnp.append(M, 1.0)
    M_new_aug = exp_aug @ M_aug
    
    return M_new_aug[:size]
