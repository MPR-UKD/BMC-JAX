"""Physical pools and scanner settings -> the static array layout JIT needs.

EVERY PHYSICAL PARAMETER IS FORCED TO FLOAT. Configs routinely write `k: 50` rather than
`k: 50.0`, and an integer reaching the solver becomes an integer array whose derivative is zero:
the parameter then cannot be fitted, and nothing says so -- the optimiser simply reports the
starting value back. Only the genuinely discrete fields (mt_lineshape, use_pulse_frame) stay
integer.

`convert_params` is duck-typed on purpose. It reads attributes and never checks a class, so any
object exposing water_pool, cest_pools, mt_pool, system and options works: bmc_jax.io's
dataclasses, a bmctool Parameters, or something of your own.
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
    water_r1 = jnp.asarray(params.water_pool.r1, dtype=float)
    water_r2 = jnp.asarray(params.water_pool.r2, dtype=float)
    water_f = jnp.asarray(params.water_pool.f, dtype=float)

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

    cest_k = jnp.asarray(k_list, dtype=float)
    cest_f = jnp.asarray(f_list, dtype=float)
    cest_dw = jnp.asarray(dw_list, dtype=float)
    cest_r1 = jnp.asarray(r1_list, dtype=float)
    cest_r2 = jnp.asarray(r2_list, dtype=float)

    # --- MT Pool ---
    if params.mt_pool is not None:
        mt_k = jnp.asarray(params.mt_pool.k, dtype=float)
        mt_f = jnp.asarray(params.mt_pool.f, dtype=float)
        mt_dw = jnp.asarray(params.mt_pool.dw, dtype=float)
        mt_r1 = jnp.asarray(params.mt_pool.r1, dtype=float)
        mt_r2 = jnp.asarray(params.mt_pool.r2, dtype=float)
        
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
    b0 = jnp.asarray(params.system.b0, dtype=float)
    gamma = jnp.asarray(params.system.gamma, dtype=float)
    b0_inhom = jnp.asarray(params.system.b0_inhom, dtype=float)
    rel_b1 = jnp.asarray(params.system.rel_b1, dtype=float)
    
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

def get_initial_magnetization(jax_params: JAXParams, scale: float = 1.0) -> jnp.ndarray:
    """Build the starting magnetisation vector.

    Parameters
    ----------
    jax_params
        The parameters the state vector must match in size and dtype.
    scale
        Longitudinal magnetisation at the start of saturation, as a fraction of equilibrium.
        1.0 is thermal equilibrium.

    Returns
    -------
    jnp.ndarray
        M0, of length 3 * (1 + n_cest) + 1.

    Notes
    -----
    `scale` below 1.0 is not a fudge factor. A sequence that reads out repeatedly does not let
    the magnetisation recover fully before the next saturation block, so saturation begins from a
    reduced state. Assuming equilibrium instead puts the whole simulated Z-spectrum too high, and
    a fit will absorb that offset into whatever parameter is free -- usually the pool size.

    Only the LONGITUDINAL components are scaled. The relaxation TARGET stays at equilibrium
    (water_f, cest_f), so the solver correctly relaxes from the reduced start back up during any
    recovery delay in the sequence. Scaling the target instead would model a smaller sample.
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

    # Scale only the longitudinal slots: water Mz and the CEST pools' Mz. The transverse
    # components are zero anyway, and the MT pool is left alone because it is not what the
    # readout depleted.
    if scale != 1.0:
        z0 = 2 * n_total_pools
        M_init = M_init.at[z0:z0 + n_total_pools].multiply(scale)

    return M_init
