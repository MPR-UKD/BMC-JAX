"""
Voxel-batched BMC simulator (NBMF-style heterogeneous-tissue path).

This module adds a SECOND batching axis to the simulator without touching the
existing physics core. The existing `simulate_spectrum` (in simulator.py)
vectorizes ONE tissue across many frequency offsets:

    vmap over offsets:   in_axes=(None, 0, None)   # params shared, events batched

This module wraps a SECOND vmap around that, mapping over a VOXEL axis on the
parameters and the initial magnetization, while sharing one event sequence:

    vmap over voxels:    in_axes=(0, None, 0)       # params per-voxel, events shared

The result is a nested vmap (voxels x offsets). The inner vmap still hands the
physics core (`bmc_matrix`, `propagate`) a single scalar tissue
at a time, so NOT A SINGLE LINE of the physics needs to change. XLA fuses both
maps into one kernel.

You ALTERNATE between the two paths simply by calling the function that fits the
task -- no mode flag, no rewrite:

  * many offsets, ONE tissue            -> simulator.simulate_spectrum   (offset axis)
  * ONE sequence, MANY tissues/voxels   -> voxels.simulate_voxels      (voxel axis)

Requirements for the voxel path:
  * Every SimParams field carries a leading voxel axis of length N_voxels
    (e.g. water_r1 has shape (N_voxels,) instead of being a scalar).
    Use `stack_params` to build this from a list of per-voxel SimParams.
  * M_init has shape (N_voxels, vector_size).
"""

from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from .params import SimParams
from .events import SimEvent
from .simulate import simulate_sequence, simulate_spectrum


# ---------------------------------------------------------------------------
# Helper: stack a list of per-voxel SimParams into one batched SimParams
# ---------------------------------------------------------------------------
def stack_params(params_list: List[SimParams]) -> SimParams:
    """
    Combine N per-voxel SimParams into a single SimParams whose every field has a
    leading voxel axis of length N.

    Example:
        single voxel:  water_r1 -> shape ()      (scalar)
        stacked (N=3): water_r1 -> shape (3,)

    This is exactly the layout the voxel `vmap` (in_axes=0) expects.

    Args:
        params_list: list of SimParams, one per voxel. They must all share the
                     same CEST-pool padding (same max_cest_pools) so their array
                     fields have matching shapes.

    Returns:
        SimParams with a leading voxel axis on every field.
    """
    if not params_list:
        raise ValueError("params_list is empty")

    # jax.tree.map stacks each leaf across the list, field by field, so the
    # NamedTuple structure is preserved automatically.
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *params_list)


def tile_params(params: SimParams, n_voxels: int) -> SimParams:
    """
    Convenience for testing: replicate ONE tissue into a voxel batch of size
    n_voxels (every voxel identical). Useful to verify that the voxel path
    reproduces the single-tissue result before feeding real heterogeneous data.
    """
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (n_voxels,) + jnp.shape(leaf)),
        params,
    )


# ---------------------------------------------------------------------------
# Voxel-batched simulation (nested vmap: voxels x offsets)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=["return_full_trajectory"])
def simulate_voxels(
    params_batch: SimParams,        # every field: (N_voxels, ...)
    events: SimEvent,               # ONE shared sequence: (n_offsets, n_events)
    M_init_batch: jnp.ndarray,      # (N_voxels, vector_size)
    return_full_trajectory: bool = False,
) -> jnp.ndarray:
    """
    Simulate MANY voxels (heterogeneous tissues) against ONE shared sequence.

    Outer vmap maps over the voxel axis of params and M_init; events are shared.
    Inside, each voxel runs the existing offset-vectorized `simulate_sequence`
    path, so the physics core is reused unchanged.

    Args:
        params_batch: SimParams with a leading voxel axis on every field.
        events: SimEvent for one sequence (shared across all voxels), shape
                (n_offsets, n_events) per field -- same object you already pass to
                `simulate_spectrum`.
        M_init_batch: initial magnetization per voxel, shape (N_voxels, vec_size).
        return_full_trajectory: forwarded to the inner simulation.

    Returns:
        If return_full_trajectory:
            (N_voxels, n_offsets, n_events, vec_size)
        else:
            (N_voxels, n_offsets, vec_size)   -- final state per offset per voxel
    """
    # Inner function = the EXISTING offset-batched simulator, one tissue at a time.
    inner = partial(
        simulate_spectrum,
        return_full_trajectory=return_full_trajectory,
    )

    # Outer vmap over voxels:
    #   params_batch -> axis 0  (one tissue per voxel)
    #   events       -> None    (shared sequence)
    #   M_init_batch -> axis 0  (one initial state per voxel)
    return jax.vmap(inner, in_axes=(0, None, 0))(
        params_batch, events, M_init_batch
    )


# ---------------------------------------------------------------------------
# Optional: extract Z-spectra (ADC samples) from the trajectory
# ---------------------------------------------------------------------------
def extract_z_spectra(
    trajectory: jnp.ndarray,
    water_z_index: int,
) -> jnp.ndarray:
    """
    Pull the water Z-magnetization at each offset out of a voxel-batched result.

    This assumes the simulator was called with return_full_trajectory=False, so
    `trajectory` has shape (N_voxels, n_offsets, vec_size) and the last axis is
    the magnetization vector. The water Z-component index is layout-dependent
    (2 * n_total_pools in the current simulation.params convention) -- pass it in so
    this helper stays decoupled from the indexing logic.

    Returns:
        Z-spectra of shape (N_voxels, n_offsets).
    """
    return trajectory[..., water_z_index]
