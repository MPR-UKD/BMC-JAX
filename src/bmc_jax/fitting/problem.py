"""One thing to simulate: parameters, a sequence, a starting state, and where to read water.

A `Problem` is what the fitter compares against data. Bundling these four together removes the
`water_z` argument that every caller previously had to compute for itself -- an index into the
solver's state layout, which callers should not have to know.

`Problem.stack` is what makes multi-experiment fits ordinary. QUESP varies saturation power,
QUEST varies saturation time, and both are the same operation: simulate several sequences with
one set of parameters and fit against all the data at once. That is not a convenience. A single
saturation level constrains the exchange FLUX k*f and not its two factors, so k and f are only
identifiable across a ladder.
"""
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from ..jax_solver.simulator import EventType, SimEvent, simulate_spectrum


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


@dataclass(frozen=True)
class Problem:
    """Parameters, a sequence, a starting state, and the index of water's Mz.

    Attributes
    ----------
    params
        A `JAXParams`, either for one sample or with a leading batch axis.
    events
        Batched `SimEvent`, shape (n_offsets, n_events).
    m0
        Initial magnetisation vector, or a batch of them.
    mz_loc
        Index of water Mz in the state vector. `Parameters.mz_loc` computes it from a config.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> prob = Problem(params, events, m0, mz_loc=4)
    >>> z = prob.simulate()
    """

    params: object
    events: SimEvent
    m0: jnp.ndarray
    mz_loc: int

    @property
    def n_offsets(self) -> int:
        """How many saturation offsets this problem simulates."""
        return int(self.events.dt.shape[0])

    @property
    def n_events(self) -> int:
        """How many sequence events each offset carries, after any padding."""
        return int(self.events.dt.shape[1])

    def simulate(self) -> jnp.ndarray:
        """Simulate and return the Z-spectrum: water Mz at every offset."""
        return simulate_spectrum(self.params, self.events, self.m0,
                                 return_full_trajectory=False)[:, self.mz_loc]

    @classmethod
    def stack(cls, problems: Sequence["Problem"]) -> "Problem":
        """Join several sequences into one, to be fitted against concatenated data.

        Parameters
        ----------
        problems
            Problems sharing parameters, starting state and `mz_loc`, differing only in their
            sequence. A QUESP ladder is one per saturation power; a QUEST ladder is one per
            saturation time.

        Returns
        -------
        Problem
            A single problem whose offset axis is every input's offsets, in order. Shorter
            sequences are padded to the longest with leading no-ops, so ladders whose events
            differ in number -- as a QUEST ladder's always do -- stack like any other.

        Raises
        ------
        ValueError
            If the list is empty or the problems disagree on `mz_loc`.

        Notes
        -----
        The measured data must be concatenated in the SAME order, which is the caller's job and
        the easiest thing to get wrong. Getting it wrong does not raise; it produces a fit.
        """
        problems = list(problems)
        if not problems:
            raise ValueError("stack needs at least one problem")
        locs = {p.mz_loc for p in problems}
        if len(locs) > 1:
            raise ValueError(f"problems disagree on mz_loc: {sorted(locs)}; they must describe "
                             "the same pools in the same order")

        width = max(p.n_events for p in problems)
        padded = [pad_events(p.events, width) for p in problems]
        events = jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *padded)
        first = problems[0]
        return cls(first.params, events, first.m0, first.mz_loc)
