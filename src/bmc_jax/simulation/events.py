"""The sequence as the simulator sees it: one SimEvent per time step, batched over offsets.

`SimEvent` and `EventType` (formerly in simulator.py), `pad_events` (formerly in fitting/experiment.py)
and `cast_events` / `repeat_events` (formerly in utils.py) live together here since 2026-10-07:
all of them are about events, and none of them needs the physics.
"""
from typing import NamedTuple

import jax.numpy as jnp


class EventType:
    """Enum-like integer constants for event types."""
    DELAY = 0
    RF_PULSE = 1
    GRADIENT = 2  # Treated as delay + spoiling
    ADC = 3
    SPOILER = 4   # Explicit spoiling event

class SimEvent(NamedTuple):
    """
    Represents a single time step or event in the simulation.
    
    This structure is designed to be stacked into a large array for jax.lax.scan,
    allowing efficient processing of the entire sequence on GPU/TPU.
    """
    type: jnp.ndarray         # EventType (int32)
    dt: jnp.ndarray           # Duration [s] (float32)
    rf_amp: jnp.ndarray       # RF Amplitude [Hz] (float32)
    rf_phase: jnp.ndarray     # RF Phase [rad] (float32)
    rf_freq: jnp.ndarray      # RF Frequency Offset [Hz] (float32)
    adc_idx: jnp.ndarray      # Index for ADC output (-1 if not ADC) (int32)

def pad_events(events: SimEvent, width: int) -> SimEvent:
    """Pad a sequence to `width` events with leading no-ops, so it can stack with longer ones.

    Parameters
    ----------
    events
        Batched events, shape (n_offsets, n_events).
    width
        Target number of events. Must be at least the current number.

    Returns
    -------
    SimEvent
        Events of shape (n_offsets, width), physically identical to the input.

    Raises
    ------
    ValueError
        If `width` is smaller than the sequence already is.

    Notes
    -----
    The padding is zero-duration DELAY events, which are a true no-op: the solver skips evolution
    when ``dt <= 1e-9``, and DELAY triggers neither the spoiler branch nor the ADC reset. Nothing
    about the simulated magnetisation changes.

    They go at the FRONT, and that is not a matter of taste. `simulate_sequence` returns
    ``final_state.last_M``, which every step overwrites, while an ADC with ``reset_after_adc``
    sets the running magnetisation back to its initial value. A no-op appended AFTER the readout
    would therefore copy that reset value into ``last_M`` and the function would return the
    starting magnetisation instead of the saturated one -- silently, with a plausible-looking
    spectrum. Padding before the sequence begins cannot do this, because the state it passes
    through is the initial state either way.

    The other tempting mistake is padding with a nonzero-duration, zero-amplitude event. That is
    not a no-op either: it is free relaxation, and it would add recovery time to exactly the
    shortest saturation in a QUEST ladder, which is the rung that carries the rate information.
    """
    n_offsets, n_events = events.dt.shape
    if width < n_events:
        raise ValueError(f"cannot pad {n_events} events down to {width}")
    if width == n_events:
        return events

    n_pad = width - n_events

    def prepend(leaf, fill):
        block = jnp.full((n_offsets, n_pad), fill, leaf.dtype)
        return jnp.concatenate([block, leaf], axis=1)

    return SimEvent(
        type=prepend(events.type, EventType.DELAY),
        dt=prepend(events.dt, 0.0),
        rf_amp=prepend(events.rf_amp, 0.0),
        rf_phase=prepend(events.rf_phase, 0.0),
        rf_freq=prepend(events.rf_freq, 0.0),
        adc_idx=prepend(events.adc_idx, -1),
    )


def cast_events(events: SimEvent, dtype) -> SimEvent:
    """Cast floating point fields of SimEvent to dtype."""
    return SimEvent(
        type=events.type,
        dt=events.dt.astype(dtype),
        rf_amp=events.rf_amp.astype(dtype),
        rf_phase=events.rf_phase.astype(dtype),
        rf_freq=events.rf_freq.astype(dtype),
        adc_idx=events.adc_idx
    )


def repeat_events(events: SimEvent, n_reps: int) -> SimEvent:
    """Replicate the batch of events n_reps times."""
    return SimEvent(
        type=jnp.tile(events.type, (n_reps, 1)),
        dt=jnp.tile(events.dt, (n_reps, 1)),
        rf_amp=jnp.tile(events.rf_amp, (n_reps, 1)),
        rf_phase=jnp.tile(events.rf_phase, (n_reps, 1)),
        rf_freq=jnp.tile(events.rf_freq, (n_reps, 1)),
        adc_idx=jnp.tile(events.adc_idx, (n_reps, 1))
    )
