"""
Parameter adapter for JAX-based BMC simulation.

This module provides the JAXParams container and conversion utilities to transform
standard BMCTool parameters into JAX-compatible arrays.
"""

from typing import NamedTuple, Optional, Any
import jax.numpy as jnp

class JAXParams(NamedTuple):
    """
    JAX-compatible container for BMC simulation parameters.
    
    All fields are JAX arrays to ensure compatibility with JIT compilation and vmap.
    CEST pool parameters are stored as arrays where the index corresponds to the pool.
    """
    # Water pool parameters
    water_r1: jnp.ndarray  # Longitudinal relaxation rate [Hz]
    water_r2: jnp.ndarray  # Transverse relaxation rate [Hz]
    water_f: jnp.ndarray   # Proton fraction (usually 1.0)

    # CEST pools (arrays of length num_cest_pools or max_cest_pools)
    cest_k: jnp.ndarray    # Exchange rates [Hz]
    cest_f: jnp.ndarray    # Proton fractions
    cest_dw: jnp.ndarray   # Chemical shifts [ppm]
    cest_r1: jnp.ndarray   # Longitudinal relaxation rates [Hz]
    cest_r2: jnp.ndarray   # Transverse relaxation rates [Hz]
    
    # MT pool (scalars, 0 if not present)
    mt_k: jnp.ndarray      # Exchange rate [Hz]
    mt_f: jnp.ndarray      # Proton fraction
    mt_dw: jnp.ndarray     # Chemical shift [ppm]
    mt_r1: jnp.ndarray     # Longitudinal relaxation rate [Hz]
    mt_r2: jnp.ndarray     # Transverse relaxation rate [Hz]
    mt_lineshape: jnp.ndarray # Lineshape type (0=None, 1=Lorentzian, 2=SuperLorentzian)

    # System parameters
    b0: jnp.ndarray        # Field strength [T]
    gamma: jnp.ndarray     # Gyromagnetic ratio [MHz/T]
    b0_inhom: jnp.ndarray  # B0 inhomogeneity [ppm]
    rel_b1: jnp.ndarray    # Relative B1 field (scaling factor)
    
    # Simulation Options
    use_pulse_frame: jnp.ndarray # Boolean flag (1=True, 0=False)

def convert_params(params: Any, max_cest_pools: Optional[int] = None) -> JAXParams:
    """
    Converts a bmctool Parameters object into a JAX-compatible JAXParams NamedTuple.
    
    Args:
        params: The bmctool Parameters object (or similar object with water_pool, cest_pools, etc).
        max_cest_pools: Optional integer to pad the CEST pool arrays to a fixed length.
                        This is useful for JIT compilation to maintain static shapes.
    
    Returns:
        JAXParams: The JAX-friendly parameter structure.
        
    Raises:
        ValueError: If max_cest_pools is smaller than the actual number of pools.
    """
    
    # --- Water Pool ---
    water_r1 = jnp.array(params.water_pool.r1)
    water_r2 = jnp.array(params.water_pool.r2)
    water_f = jnp.array(params.water_pool.f)

    # --- CEST Pools ---
    num_pools = len(params.cest_pools)
    target_size = max_cest_pools if max_cest_pools is not None else num_pools
    
    if target_size < num_pools:
        raise ValueError(f"max_cest_pools ({max_cest_pools}) is smaller than actual number of pools ({num_pools})")

    # Extract lists from pool objects
    k_list = [p.k for p in params.cest_pools]
    f_list = [p.f for p in params.cest_pools]
    dw_list = [p.dw for p in params.cest_pools]
    r1_list = [p.r1 for p in params.cest_pools]
    r2_list = [p.r2 for p in params.cest_pools]

    # Pad with zeros if necessary to reach target_size
    # Padding with 0 effectively disables the pool in the BMC equations
    padding = target_size - num_pools
    if padding > 0:
        k_list += [0.0] * padding
        f_list += [0.0] * padding
        dw_list += [0.0] * padding
        r1_list += [0.0] * padding
        r2_list += [0.0] * padding

    cest_k = jnp.array(k_list)
    cest_f = jnp.array(f_list)
    cest_dw = jnp.array(dw_list)
    cest_r1 = jnp.array(r1_list)
    cest_r2 = jnp.array(r2_list)

    # --- MT Pool ---
    if params.mt_pool is not None:
        mt_k = jnp.array(params.mt_pool.k)
        mt_f = jnp.array(params.mt_pool.f)
        mt_dw = jnp.array(params.mt_pool.dw)
        mt_r1 = jnp.array(params.mt_pool.r1)
        mt_r2 = jnp.array(params.mt_pool.r2)
        
        ls_str = params.mt_pool.lineshape.lower()
        if ls_str == 'lorentzian':
            mt_lineshape = jnp.array(1)
        elif ls_str == 'superlorentzian':
            mt_lineshape = jnp.array(2)
        else:
            mt_lineshape = jnp.array(0)
    else:
        # Default values if no MT pool is present
        mt_k = jnp.array(0.0)
        mt_f = jnp.array(0.0)
        mt_dw = jnp.array(0.0)
        mt_r1 = jnp.array(0.0)
        # Set R2 to 1.0 to avoid division by zero in lineshape calculation
        mt_r2 = jnp.array(1.0)
        mt_lineshape = jnp.array(0)

    # --- System ---
    b0 = jnp.array(params.system.b0)
    gamma = jnp.array(params.system.gamma)
    b0_inhom = jnp.array(params.system.b0_inhom)
    rel_b1 = jnp.array(params.system.rel_b1)
    
    # --- Options ---
    # Sequence evaluation threshold logic
    # Toggles optimization strategies based on predicted pulse discretizations.
    use_pulse_frame = jnp.array(1 if params.options.max_pulse_samples <= 200 else 0)

    return JAXParams(
        water_r1=water_r1,
        water_r2=water_r2,
        water_f=water_f,
        cest_k=cest_k,
        cest_f=cest_f,
        cest_dw=cest_dw,
        cest_r1=cest_r1,
        cest_r2=cest_r2,
        mt_k=mt_k,
        mt_f=mt_f,
        mt_dw=mt_dw,
        mt_r1=mt_r1,
        mt_r2=mt_r2,
        mt_lineshape=mt_lineshape,
        b0=b0,
        gamma=gamma,
        b0_inhom=b0_inhom,
        rel_b1=rel_b1,
        use_pulse_frame=use_pulse_frame,
    )

def get_initial_magnetization(jax_params: JAXParams) -> jnp.ndarray:
    """
    Constructs the initial magnetization vector based on the simulation parameters.
    
    Args:
        jax_params: JAXParams container.
        
    Returns:
        jnp.ndarray: The initial magnetization vector M0.
    """
    n_cest = jax_params.cest_k.shape[0]
    n_total_pools = 1 + n_cest
    size = 3 * n_total_pools + 1
    
    # Match the params' active float precision (float64 under x64, float32 when FIT_X64=0) so M0
    # and the params share a dtype -- avoids a float64-truncation warning in 32-bit mode.
    M_init = jnp.zeros(size, dtype=jax_params.water_f.dtype)
    
    # Set Water Z-magnetization to water fraction (typically 1.0)
    M_init = M_init.at[2 * n_total_pools].set(jax_params.water_f)
    
    # Set CEST pools Z-magnetization
    for i in range(n_cest):
        M_init = M_init.at[2 * n_total_pools + 1 + i].set(jax_params.cest_f[i])
        
    # Set MT pool Z-magnetization
    M_init = M_init.at[3 * n_total_pools].set(jax_params.mt_f)
    
    return M_init
