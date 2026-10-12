"""The forward model: Bloch-McConnell in homogeneous coordinates, simulated over a sequence.

Formerly `bmc_jax.jax_solver` (renamed on 2026-10-07; old -> new table in the README).
Layers, innermost first:
    params.py     SimParams -- the static array layout JIT needs -- to_sim_params, initial_state, cast_params
    events.py     SimEvent, EventType, pad_events, cast_events, repeat_events
    matrix.py     bmc_matrix (the homogeneous-coordinate system matrix), lorentzian, super_lorentzian
    expm.py       propagate: one time step, with expm_library or expm_plain (bmc_jax.settings.use_expm)
    simulate.py   simulate_sequence (scan over the events), simulate_spectrum (vmap over the offsets)
    voxels.py     simulate_voxels, tile_params, stack_params, extract_z_spectra: batched over samples
    spread.py     B0Spread, simulate_spectrum_spread: the B0 spread inside a sample, averaged over isochromats
    multigpu.py   sharded_vmap, pick_devices: a vmapped batch across every local GPU (import it directly)
"""
from .params import SimParams, to_sim_params, initial_state, cast_params
from .events import SimEvent, EventType, pad_events, cast_events, repeat_events
from .matrix import bmc_matrix, lorentzian, super_lorentzian
from .expm import propagate, expm_library, expm_plain
from .simulate import simulate_sequence, simulate_spectrum
from .voxels import simulate_voxels, tile_params, stack_params, extract_z_spectra
from .spread import B0Spread, simulate_spectrum_spread

__all__ = ["SimParams", "to_sim_params", "initial_state", "cast_params",
           "SimEvent", "EventType", "pad_events", "cast_events", "repeat_events",
           "bmc_matrix", "lorentzian", "super_lorentzian",
           "propagate", "expm_library", "expm_plain",
           "simulate_sequence", "simulate_spectrum",
           "simulate_voxels", "tile_params", "stack_params", "extract_z_spectra",
           "B0Spread", "simulate_spectrum_spread"]
