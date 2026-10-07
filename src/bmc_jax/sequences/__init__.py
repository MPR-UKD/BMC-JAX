"""How a sequence becomes SimEvent blocks: built in code (saturation.py) or read from a Pulseq .seq file
(pulseq.py, which needs pypulseq and is imported on first use).

Formerly bmc_jax.jax_solver.sequences and .seq_adapter (renamed on 2026-10-07).
"""
from .saturation import GAMMA_HZ_PER_T, GAMMA_RAD_PER_S_UT, cw_saturation, hz_per_ppm

__all__ = ["cw_saturation", "hz_per_ppm", "events_from_seq", "events_from_seq_batch"]


def __getattr__(name):
    if name in ("events_from_seq", "events_from_seq_batch", "prep_rf_simulation"):
        from . import pulseq
        return getattr(pulseq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
