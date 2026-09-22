"""
Utility functions for JAX-based BMC simulation.
Includes casting utilities to manage precision across hardware environments.
"""

import jax.numpy as jnp
import numpy as np

from .param_adapter import JAXParams
from .simulator import SimEvent


def cast_params_to_dtype(params: JAXParams, dtype) -> JAXParams:
    """Recursively cast JAXParams fields to dtype if they are floating point arrays."""
    new_fields = {}
    for field in params._fields:
        val = getattr(params, field)
        if hasattr(val, 'dtype') and (jnp.issubdtype(val.dtype, jnp.floating) or jnp.issubdtype(val.dtype, float)):
              arr = jnp.array(val)
              new_fields[field] = arr.astype(dtype)
        elif isinstance(val, (float, int)) and not isinstance(val, bool):
             arr = jnp.array(val, dtype=dtype)
             new_fields[field] = arr
        else:
             new_fields[field] = val
    return type(params)(**new_fields)


def cast_events_to_dtype(events: SimEvent, dtype) -> SimEvent:
    """Cast floating point fields of SimEvent to dtype."""
    return SimEvent(
        type=events.type,
        dt=events.dt.astype(dtype),
        rf_amp=events.rf_amp.astype(dtype),
        rf_phase=events.rf_phase.astype(dtype),
        rf_freq=events.rf_freq.astype(dtype),
        adc_idx=events.adc_idx
    )


def replicate_events(events: SimEvent, n_reps: int) -> SimEvent:
    """Replicate the batch of events n_reps times."""
    return SimEvent(
        type=jnp.tile(events.type, (n_reps, 1)),
        dt=jnp.tile(events.dt, (n_reps, 1)),
        rf_amp=jnp.tile(events.rf_amp, (n_reps, 1)),
        rf_phase=jnp.tile(events.rf_phase, (n_reps, 1)),
        rf_freq=jnp.tile(events.rf_freq, (n_reps, 1)),
        adc_idx=jnp.tile(events.adc_idx, (n_reps, 1))
    )
