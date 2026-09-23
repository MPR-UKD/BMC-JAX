"""
MRF dictionary generation on top of the voxel-batched BMC simulator.

A "dictionary entry" is exactly a voxel in voxel_simulator.py's sense: one
JAXParams instance run against one shared MRF acquisition sequence. This
module only adds what voxel_simulator.py doesn't have -- turning a
{field_name: values} sweep into the batched JAXParams that `simulate_voxels`
expects, plus chunking so a large Cartesian grid doesn't have to fit in GPU
memory in one call.
"""
from typing import Dict, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .param_adapter import JAXParams
from .simulator import SimEvent
from .voxel_simulator import simulate_voxels, extract_z_spectra


def build_param_grid(base_params: JAXParams, grid: Dict[str, Sequence[float]]) -> JAXParams:
    """
    Cartesian-product a sweep of JAXParams fields into one batched JAXParams.

    Args:
        base_params: a single-tissue JAXParams (e.g. from case_loader.load_case)
                      supplying every field NOT being swept.
        grid: {field_name: 1D array_like of values}. field_name must name a
              JAXParams field that is a scalar or a single-element array in
              base_params (e.g. "water_r1", "mt_k", or "cest_k"/"cest_f"/...
              only when there is exactly one CEST pool). Sweeping one pool of
              a multi-pool field isn't supported here -- pass the full
              per-entry array yourself if you need that.

    Returns:
        JAXParams with every field broadcast to a leading axis of length
        prod(len(v) for v in grid.values()), ready for `simulate_voxels`.
    """
    if not grid:
        raise ValueError("grid must have at least one swept parameter")

    names = list(grid.keys())
    axes = [jnp.asarray(grid[n]) for n in names]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    flat = [m.reshape(-1) for m in mesh]
    n_entries = flat[0].shape[0]

    # Broadcast every field (swept or not) to a leading (n_entries,) axis.
    batched = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (n_entries,) + jnp.shape(leaf)),
        base_params,
    )

    updates = {}
    for name, values in zip(names, flat):
        target_shape = jnp.shape(getattr(base_params, name))
        if target_shape != () and any(s != 1 for s in target_shape):
            raise ValueError(
                f"Cannot sweep '{name}': base field has shape {target_shape} "
                "(more than one element). Sweeping into a specific slot of a "
                "multi-pool field is not supported by build_param_grid."
            )
        updates[name] = values.reshape((n_entries,) + target_shape)

    return batched._replace(**updates)


def generate_dictionary(
    base_params: JAXParams,
    seq_base: SimEvent,
    M_init: jnp.ndarray,
    grid: Dict[str, Sequence[float]],
    water_z_index: int,
    chunk_size: int = 20_000,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Generate a Z-spectrum fingerprint for every entry of the Cartesian
    product of `grid`, chunked so the batch fed to `simulate_voxels` at once
    is bounded (a full grid over several parameters can be far larger than
    GPU memory allows in one vmap call).

    Args:
        base_params: single-tissue JAXParams for every field not in `grid`.
        seq_base: the shared MRF acquisition sequence (n_offsets, n_events).
        M_init: initial magnetization vector, shape (vector_size,).
        grid: sweep spec, see `build_param_grid`.
        water_z_index: index of the water Z-magnetization component in the
                       state vector (as used by `extract_z_spectra`).
        chunk_size: max number of dictionary entries per `simulate_voxels` call.

    Returns:
        entries: {field_name: np.ndarray(N,)} grid coordinates for each row,
                  aligned with `fingerprints`.
        fingerprints: np.ndarray (N, n_offsets) -- one MRF fingerprint per entry.
    """
    params_batch = build_param_grid(base_params, grid)
    n_entries = params_batch.water_r1.shape[0]
    size = M_init.shape[0]

    entries = {name: np.asarray(getattr(params_batch, name)).reshape(-1) for name in grid}
    fingerprints = np.empty((n_entries, seq_base.rf_amp.shape[0]), dtype=np.float32)

    for start in range(0, n_entries, chunk_size):
        end = min(start + chunk_size, n_entries)
        chunk = jax.tree.map(lambda leaf: leaf[start:end], params_batch)
        M_chunk = jnp.broadcast_to(M_init, (end - start, size))
        out = simulate_voxels(chunk, seq_base, M_chunk, return_full_trajectory=False)
        fingerprints[start:end] = np.asarray(extract_z_spectra(out, water_z_index))

    return entries, fingerprints
