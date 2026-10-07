"""An experiment: a sample, a sequence to measure it with, and where to read the answer.

Bundling those together removes the `water_z` argument every caller previously had to work out
for itself -- an index into the solver's state layout, which a caller should not have to know.

`Experiment.stack` is what makes ladders ordinary. QUESP varies saturation power, QUEST varies
saturation time, and both are the same operation: several sequences, one set of pools, fitted
against all the data at once. That is not a convenience. A single saturation level constrains the
exchange FLUX k*f and not its two factors, so k and f only become identifiable across a ladder.
"""
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from ..simulation.events import EventType, SimEvent, pad_events
from ..simulation.simulate import simulate_spectrum


@dataclass(frozen=True)
class Experiment:
    """Parameters, a sequence, a starting state, and the index of water's Mz.

    Attributes
    ----------
    params
        A `SimParams`, either for one sample or with a leading batch axis.
    events
        Batched `SimEvent`, shape (n_offsets, n_events).
    m0
        Initial magnetisation vector, or a batch of them.
    mz_loc
        Index of water Mz in the state vector. `Parameters.mz_loc` computes it from a config.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> quesp = Experiment.stack([Experiment(pools, ev, m0, mz_loc) for ev in ladder])
    >>> z = quesp.simulate()
    """

    params: object
    events: SimEvent
    m0: jnp.ndarray
    mz_loc: int

    @property
    def n_offsets(self) -> int:
        """How many saturation offsets this experiment simulates."""
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
    def from_parameters(cls, params, events) -> "Experiment":
        """Build from a config object and a sequence. The entry point to prefer.

        Parameters
        ----------
        params
            Anything with `water_pool`, `cest_pools`, `mt_pool`, `system` and `options` --
            `bmc_jax.io.Parameters`, a bmctool `Parameters`, or your own object.
        events
            Batched `SimEvent`, from `cw_saturation` or from a Pulseq `.seq`.

        Returns
        -------
        Experiment

        Notes
        -----
        This does the three things a caller would otherwise do by hand, and does the third one
        correctly: convert the pools, build the starting magnetisation, and find water's Mz.

        The third is the reason this exists. `options.scale` -- the magnetisation the sequence
        starts from, as a fraction of equilibrium -- is part of the config format, but the
        parameter conversion does not carry it, so a caller who builds the starting state
        themselves silently gets equilibrium no matter what their config says. Anything below 1.0
        raises the whole simulated Z-spectrum, and a fit absorbs that into the pool size.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> exp = Experiment.from_parameters(example_config("amide"), events)
        """
        from ..simulation.params import to_sim_params, initial_state

        jax_params = to_sim_params(params)
        scale = float(getattr(params.options, "scale", 1.0))
        return cls(jax_params, events, initial_state(jax_params, scale), params.mz_loc)

    @classmethod
    def stack(cls, experiments: Sequence["Experiment"]) -> "Experiment":
        """Join several sequences into one, to be fitted against concatenated data.

        Parameters
        ----------
        experiments
            Problems sharing parameters, starting state and `mz_loc`, differing only in their
            sequence. A QUESP ladder is one per saturation power; a QUEST ladder is one per
            saturation time.

        Returns
        -------
        Experiment
            A single experiment whose offset axis is every input's offsets, in order. Shorter
            sequences are padded to the longest with leading no-ops, so ladders whose events
            differ in number -- as a QUEST ladder's always do -- stack like any other.

        Raises
        ------
        ValueError
            If the list is empty or the experiments disagree on `mz_loc`.

        Notes
        -----
        The measured data must be concatenated in the SAME order, which is the caller's job and
        the easiest thing to get wrong. Getting it wrong does not raise; it produces a fit.
        """
        experiments = list(experiments)
        if not experiments:
            raise ValueError("stack needs at least one experiment")
        locs = {p.mz_loc for p in experiments}
        if len(locs) > 1:
            raise ValueError(f"experiments disagree on mz_loc: {sorted(locs)}; they must describe "
                             "the same pools in the same order")

        width = max(p.n_events for p in experiments)
        padded = [pad_events(p.events, width) for p in experiments]
        events = jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *padded)
        first = experiments[0]
        return cls(first.params, events, first.m0, first.mz_loc)
