"""
Sequence adapter to convert PyPulseq sequences to JAX-compatible SimEvents.

This module handles the parsing of PyPulseq sequence objects and transforms them
into a flat list of simulation events (SimEvent) that can be processed by the
JAX solver.
"""

import numpy as np
import pypulseq as pp
from types import SimpleNamespace
from typing import Tuple, Optional
import jax.numpy as jnp

from ..simulation.events import SimEvent, EventType

def events_from_seq(
    seq: pp.Sequence,
    max_pulse_samples: int = 500,
    max_events: Optional[int] = None
) -> Tuple[SimEvent, int]:
    """
    Convert a PyPulseq sequence into a JAX-compatible SimEvent structure.
    
    The result is a single SimEvent object where each field is an array of values,
    representing the sequence of events. This structure is optimized for 
    processing with jax.lax.scan.
    
    Args:
        seq: The PyPulseq sequence to convert.
        max_pulse_samples: Maximum number of samples to use when discretizing shaped RF pulses.
        max_events: If provided, pads the event arrays to this length. This is useful
                   for maintaining static array shapes for JIT compilation.
        
    Returns:
        events: The sequence of events as JAX arrays wrapped in a SimEvent NamedTuple.
        n_events: The actual number of events in the sequence (before padding).
        
    Raises:
        ValueError: If the sequence contains more events than max_events.
    """
    
    # Lists to collect event data
    types = []
    dts = []
    rf_amps = []
    rf_phases = []
    rf_freqs = []
    adc_idxs = []
    
    current_adc = 0
    accum_phase = 0.0

    # Iterate through sequence blocks
    block_events = seq.block_events
    
    for block_idx in range(1, len(block_events) + 1):
        block = seq.get_block(block_idx)
        
        # 1. ADC Event
        if block.adc is not None:
            # ADC events are treated as instantaneous markers for data recording
            types.append(EventType.ADC)
            dts.append(0.0)
            rf_amps.append(0.0)
            rf_phases.append(0.0)
            rf_freqs.append(0.0)
            adc_idxs.append(current_adc)
            current_adc += 1
            accum_phase = 0.0
        # 2. RF Pulse
        elif block.rf is not None:
            amp, ph, dt, delay_after = prep_rf_simulation(block, max_pulse_samples)
            
            phase_offset = float(block.rf.phase_offset)
            freq_offset = float(block.rf.freq_offset)
            current_accum = accum_phase
            phase_degree = float(dt) * amp.size * 360 * freq_offset
            accum_phase += (phase_degree % 360) / 180 * np.pi
            # Handle block pulse (single value)
            if np.isscalar(amp) or amp.size == 1:
                # Subdivide extended block pulses to ensure stable matrix exponential evaluations
                MAX_BLOCK_DT = 0.1 # 100ms max step
                total_dt = float(dt)
                
                if total_dt > MAX_BLOCK_DT:
                    n_substeps = int(np.ceil(total_dt / MAX_BLOCK_DT))
                    sub_dt = total_dt / n_substeps
                    # Orient phase to standard solver layout constraints
                    current_ph = -float(np.squeeze(ph)) + phase_offset - current_accum
                    freq = float(np.squeeze(block.rf.freq_offset))
                    
                    for _ in range(n_substeps):
                        types.append(EventType.RF_PULSE)
                        dts.append(sub_dt)
                        rf_amps.append(float(np.squeeze(amp)))
                        rf_phases.append(current_ph)
                        rf_freqs.append(freq)
                        adc_idxs.append(-1)
                        
                        # Preserve frame-dependent phase shifts across continuous sub-steps
                        # Explicit accumulation relies entirely on explicit pulse phase variables.
                        pass
                else:
                    types.append(EventType.RF_PULSE)
                    dts.append(total_dt)
                    rf_amps.append(float(np.squeeze(amp)))
                    # Orient phase to standard solver layout constraints
                    rf_phases.append(-float(np.squeeze(ph)) + phase_offset - current_accum)
                    freq = float(np.squeeze(block.rf.freq_offset))
                    rf_freqs.append(freq)
                    adc_idxs.append(-1)
            else:
                # Handle shaped pulse by breaking into small steps
                n_steps = amp.size
                for i in range(n_steps):
                    types.append(EventType.RF_PULSE)
                    dts.append(float(dt)) # dt is the step size here
                    rf_amps.append(float(np.squeeze(amp[i])))
                    # Orient phase to standard solver layout constraints
                    rf_phases.append(-float(np.squeeze(ph[i])) + phase_offset - current_accum)
                    rf_freqs.append(float(np.squeeze(block.rf.freq_offset)))
                    adc_idxs.append(-1)
            
            # Handle delay after pulse if any
            if delay_after > 1e-12:
                types.append(EventType.DELAY)
                dts.append(float(delay_after))
                rf_amps.append(0.0)
                rf_phases.append(0.0)
                rf_freqs.append(0.0)
                adc_idxs.append(-1)

        # 3. Spoiler / Gradient
        elif block.gz is not None:
            # Gradients are treated as explicit spoiling events
            types.append(EventType.SPOILER) 
            dts.append(float(block.block_duration))
            rf_amps.append(0.0)
            rf_phases.append(0.0)
            rf_freqs.append(0.0)
            adc_idxs.append(-1)

        # 4. Delay
        elif hasattr(block, 'block_duration') and block.block_duration != '0':
            types.append(EventType.DELAY)
            dts.append(float(block.block_duration))
            rf_amps.append(0.0)
            rf_phases.append(0.0)
            rf_freqs.append(0.0)
            adc_idxs.append(-1)
            
    # Convert to numpy arrays
    n_actual = len(types)
    
    if max_events is not None:
        if n_actual > max_events:
            raise ValueError(f"Sequence has {n_actual} events, but max_events is {max_events}")
        
        # Pad with zero-duration delays
        pad_len = max_events - n_actual
        types.extend([EventType.DELAY] * pad_len)
        dts.extend([0.0] * pad_len)
        rf_amps.extend([0.0] * pad_len)
        rf_phases.extend([0.0] * pad_len)
        rf_freqs.extend([0.0] * pad_len)
        adc_idxs.extend([-1] * pad_len)
    
    # Create SimEvent with JAX-compatible arrays
    # Use float32 for speed if x64 not enabled
    events = SimEvent(
        type=jnp.array(types, dtype=jnp.int32),
        dt=jnp.array(dts, dtype=jnp.float32),
        rf_amp=jnp.array(rf_amps, dtype=jnp.float32),
        rf_phase=jnp.array(rf_phases, dtype=jnp.float32),
        rf_freq=jnp.array(rf_freqs, dtype=jnp.float32),
        adc_idx=jnp.array(adc_idxs, dtype=jnp.int32)
    )
    
    return events, n_actual

def events_from_seq_batch(
    seq: pp.Sequence,
    max_pulse_samples: int = 500
) -> Tuple[SimEvent, int]:
    """
    Convert a PyPulseq sequence into a batch of JAX-compatible SimEvents.
    
    Splits the sequence into segments, where each segment ends with an ADC event.
    This allows parallel simulation of all offsets using jax.vmap.
    
    Args:
        seq: The PyPulseq sequence to convert.
        max_pulse_samples: Maximum number of samples to use when discretizing shaped RF pulses.
        
    Returns:
        events: SimEvent object where each field is a 2D array (n_offsets, max_events_per_offset).
        n_batches: The number of batches (offsets).
    """
    
    batch_events_list = []
    
    # Current batch accumulators
    types = []
    dts = []
    rf_amps = []
    rf_phases = []
    rf_freqs = []
    adc_idxs = []
    
    current_adc = 0
    accum_phase = 0.0
    
    block_events = seq.block_events
    
    for block_idx in range(1, len(block_events) + 1):
        block = seq.get_block(block_idx)
        
        # 1. ADC Event
        if block.adc is not None:
            types.append(EventType.ADC)
            dts.append(0.0)
            rf_amps.append(0.0)
            rf_phases.append(0.0)
            rf_freqs.append(0.0)
            adc_idxs.append(current_adc)
            current_adc += 1
            accum_phase = 0.0
            # END OF BATCH
            batch_events_list.append({
                'type': types,
                'dt': dts,
                'rf_amp': rf_amps,
                'rf_phase': rf_phases,
                'rf_freq': rf_freqs,
                'adc_idx': adc_idxs
            })
            
            # Reset accumulators
            types = []
            dts = []
            rf_amps = []
            rf_phases = []
            rf_freqs = []
            adc_idxs = []
            current_adc = 0
            
        # 2. RF Pulse
        elif block.rf is not None:
            amp, ph, dt, delay_after = prep_rf_simulation(block, max_pulse_samples)
            
            phase_offset = float(block.rf.phase_offset)
            freq_offset = float(block.rf.freq_offset)
            current_accum = accum_phase
            phase_degree = float(dt) * amp.size * 360 * freq_offset
            accum_phase += (phase_degree % 360) / 180 * np.pi
            # Handle block pulse (single value)
            if np.isscalar(amp) or amp.size == 1:
                # Subdivide extended block pulses to ensure stable matrix exponential evaluations
                MAX_BLOCK_DT = 0.1 # 100ms max step
                total_dt = float(dt)
                
                if total_dt > MAX_BLOCK_DT:
                    n_substeps = int(np.ceil(total_dt / MAX_BLOCK_DT))
                    sub_dt = total_dt / n_substeps
                    # Orient phase to standard solver layout constraints
                    current_ph = -float(np.squeeze(ph)) + phase_offset - current_accum
                    freq = float(np.squeeze(block.rf.freq_offset))
                    
                    for _ in range(n_substeps):
                        types.append(EventType.RF_PULSE)
                        dts.append(sub_dt)
                        rf_amps.append(float(np.squeeze(amp)))
                        rf_phases.append(current_ph)
                        rf_freqs.append(freq)
                        adc_idxs.append(-1)
                else:
                    types.append(EventType.RF_PULSE)
                    dts.append(total_dt)
                    rf_amps.append(float(np.squeeze(amp)))
                    # Orient phase to standard solver layout constraints
                    rf_phases.append(-float(np.squeeze(ph)) + phase_offset - current_accum)
                    freq = float(np.squeeze(block.rf.freq_offset))
                    rf_freqs.append(freq)
                    adc_idxs.append(-1)
            else:
                # Handle shaped pulse by breaking into small steps
                n_steps = amp.size
                for i in range(n_steps):
                    types.append(EventType.RF_PULSE)
                    dts.append(float(dt)) # dt is the step size here
                    rf_amps.append(float(np.squeeze(amp[i])))
                    # Orient phase to standard solver layout constraints
                    rf_phases.append(-float(np.squeeze(ph[i])) + phase_offset - current_accum)
                    rf_freqs.append(float(np.squeeze(block.rf.freq_offset)))
                    adc_idxs.append(-1)
            
            # Handle delay after pulse if any
            if delay_after > 1e-12:
                types.append(EventType.DELAY)
                dts.append(float(delay_after))
                rf_amps.append(0.0)
                rf_phases.append(0.0)
                rf_freqs.append(0.0)
                adc_idxs.append(-1)

        # 3. Spoiler / Gradient
        elif block.gz is not None:
            # Gradients are treated as explicit spoiling events
            types.append(EventType.SPOILER) 
            dts.append(float(block.block_duration))
            rf_amps.append(0.0)
            rf_phases.append(0.0)
            rf_freqs.append(0.0)
            adc_idxs.append(-1)

        # 4. Delay
        elif hasattr(block, 'block_duration') and block.block_duration != '0':
            types.append(EventType.DELAY)
            dts.append(float(block.block_duration))
            rf_amps.append(0.0)
            rf_phases.append(0.0)
            rf_freqs.append(0.0)
            adc_idxs.append(-1)
            
    # Pad batches
    if not batch_events_list:
        # Handle empty sequence case
        max_len = 0
        n_batches = 0
    else:
        max_len = max([len(b['type']) for b in batch_events_list])
        n_batches = len(batch_events_list)
    
    # Initialize arrays
    # Shape: (n_batches, max_len)
    type_arr = np.zeros((n_batches, max_len), dtype=np.int32) # Default 0 (DELAY)
    dt_arr = np.zeros((n_batches, max_len), dtype=np.float64)
    rf_amp_arr = np.zeros((n_batches, max_len), dtype=np.float64)
    rf_phase_arr = np.zeros((n_batches, max_len), dtype=np.float64)
    rf_freq_arr = np.zeros((n_batches, max_len), dtype=np.float64)
    adc_idx_arr = np.full((n_batches, max_len), -1, dtype=np.int32)
    
    for i, b in enumerate(batch_events_list):
        l = len(b['type'])
        type_arr[i, :l] = b['type']
        dt_arr[i, :l] = b['dt']
        rf_amp_arr[i, :l] = b['rf_amp']
        rf_phase_arr[i, :l] = b['rf_phase']
        rf_freq_arr[i, :l] = b['rf_freq']
        adc_idx_arr[i, :l] = b['adc_idx']
        
    events = SimEvent(
        type=jnp.array(type_arr, dtype=jnp.int32),
        dt=jnp.array(dt_arr),
        rf_amp=jnp.array(rf_amp_arr),
        rf_phase=jnp.array(rf_phase_arr),
        rf_freq=jnp.array(rf_freq_arr),
        adc_idx=jnp.array(adc_idx_arr, dtype=jnp.int32)
    )
    
    return events, n_batches

def prep_rf_simulation(
    block: SimpleNamespace,
    max_pulse_samples: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Resample amplitude and phase of an RF event within a given block.
    
    This function handles the discretization of shaped pulses and calculation
    of timing parameters. Adapted from the original BMCTool implementation.
    
    Args:
        block: The PyPulseq block containing the RF event.
        max_pulse_samples: Maximum number of samples for discretization.
        
    Returns:
        Tuple containing:
        - amp_: Resampled amplitude array or scalar
        - ph_: Resampled phase array or scalar
        - dtp_: Time step duration
        - delay_after_pulse: Remaining duration in the block after the pulse
    """
    # Get amplitude and phase of RF pulse
    amp = np.abs(block.rf.signal)
    ph = np.angle(block.rf.signal)

    # Find all non-zero sample points
    idx = np.argwhere(amp > 1e-6)

    # Calculate time step and delay after pulse
    rf_length = amp.size
    dtp = block.rf.t[1] - block.rf.t[0]
    delay_after_pulse = (rf_length - idx.size) * dtp

    # Remove all zero samples
    amp = amp[idx]
    ph = ph[idx]

    # Get maximum number of unique samples in amplitude and/or phase
    n_unique = max(np.unique(amp).size, np.unique(ph).size)

    # Block pulse for seq-files >= 1.4.0
    if n_unique == 1 and amp.size == 2:
        amp_ = amp[0]
        ph_ = ph[0]
        dtp_ = dtp
    # Block pulse for seq-files < 1.4.0
    elif n_unique == 1:
        amp_ = amp[0]
        ph_ = ph[0]
        dtp_ = dtp * amp.size
    # Shaped pulse
    elif amp.size > max_pulse_samples:
        sample_factor = int(np.ceil(amp.size / max_pulse_samples))
        amp_ = np.squeeze(amp[::sample_factor])
        ph_ = np.squeeze(ph[::sample_factor])
        dtp_ = dtp * sample_factor
    else:
        amp_ = np.squeeze(amp)
        ph_ = np.squeeze(ph)
        dtp_ = dtp

    return amp_, ph_, dtp_, delay_after_pulse

