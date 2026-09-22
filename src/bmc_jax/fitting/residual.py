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

from ..jax_solver.simulator import simulate_spectrum
from ..jax_solver.voxel_simulator import simulate_voxels, tile_params, extract_z_spectra
from ..jax_solver.utils import cast_params_to_dtype, cast_events_to_dtype
from .spec import apply_x

def _cast_problem(base, events, M0, dtype):
    """Downcast the whole problem (params, sequence, init state) to a dtype."""
    if dtype is None:
        return base, events, M0
    return cast_params_to_dtype(base, dtype), cast_events_to_dtype(events, dtype), M0.astype(dtype)


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
