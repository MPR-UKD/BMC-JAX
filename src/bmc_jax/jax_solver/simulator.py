"""
Core JAX-based BMC Simulator.

This module implements the main simulation loop using JAX primitives. It defines
the state transitions and event handling logic for the Bloch-McConnell equations.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from .param_adapter import JAXParams
from .solver import build_bmc_matrix_and_vector, solve_step

class SimState(NamedTuple):
    """State carried through the simulation scan."""
    M: jnp.ndarray           # Current magnetization vector
    last_M: jnp.ndarray      # Magnetization vector from the last step (before reset)

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

def simulate_step(state: SimState, event: SimEvent, params: JAXParams) -> Tuple[SimState, jnp.ndarray]:
    """
    Step function for jax.lax.scan that evolves the system state based on a single event.
    
    Args:
        state: Current simulation state (Magnetization and Phase).
        event: The event to simulate (RF, Delay, ADC, etc.).
        params: Physical parameters of the system.
        
    Returns:
        Tuple of (new_state, output_magnetization).
    """
    M = state.M
    
    # --- 1. RF / Delay Simulation ---
    # Use pulse phase directly
    current_phase = event.rf_phase
    
    # Build Bloch-McConnell Matrix and Vector
    A, C = build_bmc_matrix_and_vector(params, event.rf_amp, current_phase, event.rf_freq)
    
    # Define evolution functions for conditional execution
    def evolve(m, dt, a, c):
        return solve_step(m, dt, a, c)
    
    def no_evolve(m, dt, a, c):
        return m

    # Evolve magnetization if duration > 0
    M_new = jax.lax.cond(event.dt > 1e-9, evolve, no_evolve, M, event.dt, A, C)
    
    # --- 2. Special Handling based on Event Type ---
    
    def apply_spoiler(m):
        """Zero out transverse magnetization components."""
        n_cest = params.cest_k.shape[0]
        n_total = 1 + n_cest # Water + CEST pools
        
        mask = jnp.ones_like(m)
        
        # Zero out X components (indices 0 to n_total-1)
        mask = mask.at[0:n_total].set(0.0)
        
        # Zero out Y components (indices n_total to 2*n_total-1)
        mask = mask.at[n_total:2*n_total].set(0.0)
        
        return m * mask

    # Apply spoiler if type is GRADIENT (2) or SPOILER (4)
    is_spoiler = (event.type == EventType.GRADIENT) | (event.type == EventType.SPOILER)
    M_new = jax.lax.cond(is_spoiler, apply_spoiler, lambda x: x, M_new)
    
    # Store M_new as last_M (before any reset happens in the caller)
    return SimState(M=M_new, last_M=M_new), M_new


from functools import partial

def simulate_sequence(params: JAXParams, events: SimEvent, M_init: jnp.ndarray, reset_after_adc: bool = True, return_full_trajectory: bool = True) -> jnp.ndarray:
    """
    Simulate the entire sequence using jax.lax.scan.
    
    Args:
        params: JAXParams object containing physical parameters.
        events: SimEvent object containing arrays of event data.
        M_init: Initial magnetization vector.
        reset_after_adc: Whether to reset magnetization to M_init after each ADC event.
        return_full_trajectory: If True, returns (N_events, Vector_Size). 
                                If False, returns (Vector_Size,) [Final State].
        
    Returns:
        Trajectory or Final State.
    """
    
    # Define the step function with closure over params and M_init
    @jax.checkpoint
    def step_fn(state, event):
        # Run the physics step
        new_state, m_out = simulate_step(state, event, params)
        
        # Handle Reset Logic
        # If the current event is an ADC and reset is enabled, the state for the 
        # NEXT iteration is reset to initial conditions.
        def do_reset(s):
            # Keep last_M from new_state, but reset M
            return SimState(M=M_init, last_M=s.last_M)
            
        final_state = jax.lax.cond(
            (event.type == EventType.ADC) & reset_after_adc,
            do_reset,
            lambda s: s,
            new_state
        )
        
        # Memory optimization structure.
        # Bypass deep trajectory retention when localized outputs suffice.
        output = m_out if return_full_trajectory else None
        
        return final_state, output

    # Initialize State
    init_state = SimState(M=M_init, last_M=M_init)
    
    # Run Scan
    final_state, trajectory = jax.lax.scan(step_fn, init_state, events)
    
    if return_full_trajectory:
        return trajectory
    else:
        # Return the last valid magnetization (before reset)
        return final_state.last_M

@partial(jax.jit, static_argnames=['return_full_trajectory'])
def simulate_spectrum(params: JAXParams, events: SimEvent, M_init: jnp.ndarray, return_full_trajectory: bool = True) -> jnp.ndarray:
    """
    Simulate a batch of sequences in parallel using jax.vmap.
    
    Args:
        params: JAXParams object.
        events: SimEvent object where each field is (N_batches, N_events).
        M_init: Initial magnetization vector.
        return_full_trajectory: Whether to return full history or just final state.
        
    Returns:
        Trajectory (N_batches, N_events, Vector_Size) or Final State (N_batches, Vector_Size).
    """
    return jax.vmap(partial(simulate_sequence, return_full_trajectory=return_full_trajectory), in_axes=(None, 0, None))(params, events, M_init)
