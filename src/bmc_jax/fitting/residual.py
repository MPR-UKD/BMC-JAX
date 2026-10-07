"""residual.py -- turn (parameters, sequence, measurement) into the vector LM minimises.

A residual, not a scalar loss. Levenberg-Marquardt needs the vector of per-offset differences
so it can form J^T J; handing it a summed mean-square would throw away exactly the structure it
exists to exploit.

`skip_first` drops the first simulated point. Scanner sequences usually acquire an unsaturated
M0 image before the ladder, and comparing it against a saturated model point is meaningless --
the same "drop the first sample" rule that applies to every recovery measurement.
"""
import jax
import jax.numpy as jnp

from ..simulation.simulate import simulate_spectrum
from ..simulation.voxels import simulate_voxels, tile_params, extract_z_spectra
from ..simulation.params import cast_params
from ..simulation.events import cast_events
from .params import write_params

def _cast_problem(base, events, M0, dtype):
    """Downcast the whole problem (params, sequence, init state) to a dtype."""
    if dtype is None:
        return base, events, M0
    return cast_params(base, dtype), cast_events(events, dtype), M0.astype(dtype)


def residual_one(base, events, M0, water_z, spec, z_meas, skip_first=True):
    """Returns r(x_scaled) -> residual vector for ONE voxel (z_sim - z_meas)."""
    z_t = jnp.asarray(z_meas[1:] if skip_first else z_meas)

    def resid(x_scaled):
        params = write_params(base, x_scaled, spec)
        z = simulate_spectrum(params, events, M0, return_full_trajectory=False)[:, water_z]
        zc = z[1:] if skip_first else z
        return zc - z_t
    return resid


def residual_batch(base, events, M0, water_z, spec, z_meas_batch, skip_first=True):
    """Returns R(X_scaled) -> (N_voxels, n_resid) residuals for the whole phantom.
    base is ONE shared SimParams (fixed fields); X_scaled is (N_voxels, n_params)."""
    Z_t = jnp.asarray(z_meas_batch[:, 1:] if skip_first else z_meas_batch)
    n_vox = z_meas_batch.shape[0]
    base_b = tile_params(base, n_vox)                       # broadcast fixed fields

    def resid(X_scaled):
        params_b = jax.vmap(write_params, in_axes=(0, 0, None))(base_b, X_scaled, spec)
        Z = simulate_voxels(params_b, events, M0, return_full_trajectory=False)
        Zc = extract_z_spectra(Z, water_z)                  # (N, n_offsets)
        Zc = Zc[:, 1:] if skip_first else Zc
        return Zc - Z_t                                     # (N, n_resid)
    return resid
