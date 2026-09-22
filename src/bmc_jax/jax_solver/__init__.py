"""The solver: parameters in, Z-spectrum out.

There was no __init__.py here before -- this directory was imported by putting it directly on
sys.path, which is exactly the arrangement that let five copies of it drift apart. As a package
it can only be reached through `bmc_jax`, so there is one copy and the import says where it came
from.

Layers, innermost first:
    solver.py           builds the homogeneous-coordinate BMC matrix and takes one time step
    simulator.py        scans that step over a sequence; vmaps it over the offsets
    voxel_simulator.py  the same, batched over voxels/samples with per-voxel parameters
    param_adapter.py    physical pools -> JAXParams, the static array layout JIT needs
    seq_adapter.py      a Pulseq .seq -> padded SimEvent blocks of fixed shape
    sharded.py          spreads an already-vmapped batch across every local GPU
"""
from .param_adapter import JAXParams, convert_params, get_initial_magnetization
from .simulator import SimEvent, EventType, simulate_sequence, simulate_spectrum
from .voxel_simulator import simulate_voxels, tile_params, extract_z_spectra
from .utils import cast_params_to_dtype, cast_events_to_dtype

__all__ = ["JAXParams", "convert_params", "get_initial_magnetization",
           "SimEvent", "EventType", "simulate_sequence", "simulate_spectrum",
           "simulate_voxels", "tile_params", "extract_z_spectra",
           "cast_params_to_dtype", "cast_events_to_dtype"]
