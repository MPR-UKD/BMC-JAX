"""Stacking several sequences into one fit, and the padding that makes it possible.

The load-bearing test here is `test_padding_at_the_end_would_destroy_the_answer`. It builds the
wrong padding deliberately and shows what it does, so the reason the code pads at the front is
recorded as evidence rather than as a claim in a comment.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmc_jax.fitting import Experiment
from bmc_jax.simulation import pad_events
from bmc_jax.io import example_config
from bmc_jax.simulation import EventType, SimEvent, to_sim_params, initial_state, simulate_spectrum
from bmc_jax.sequences import cw_saturation

PPM = np.linspace(-5.0, 5.0, 21)


def experiment_at(tsat_s, b1_uT=1.0, name="amide"):
    """One CW saturation experiment. Longer tsat does NOT mean more events here -- see below."""
    p = example_config(name)
    jp = to_sim_params(p)
    ev = cw_saturation(PPM, b1_uT=b1_uT, tsat_s=tsat_s, b0_T=p.system.b0)
    return Experiment(jp, ev, initial_state(jp), p.mz_loc)


def test_front_padding_changes_nothing():
    """Zero-duration DELAYs before the sequence are a true no-op, to the last bit."""
    exp_ = experiment_at(2.0)
    z_plain = np.asarray(exp_.simulate())
    padded = Experiment(exp_.params, pad_events(exp_.events, exp_.n_events + 37),
                     exp_.m0, exp_.mz_loc)
    assert padded.n_events == exp_.n_events + 37
    assert np.abs(z_plain - np.asarray(padded.simulate())).max() == 0.0


def test_padding_at_the_end_would_destroy_the_answer():
    """Why the padding goes at the front, demonstrated rather than asserted.

    `simulate_sequence` returns `final_state.last_M`, which every step overwrites, and an ADC
    with `reset_after_adc` sets the running magnetisation back to its initial value. A no-op
    appended AFTER the readout therefore copies that reset value into `last_M`, and the function
    returns the STARTING magnetisation instead of the saturated one.

    The failure is silent: the result is a smooth, plausible, completely flat Z-spectrum.
    """
    exp_ = experiment_at(2.0)
    z_correct = np.asarray(exp_.simulate())

    n_off, n_ev = exp_.events.dt.shape

    def append(leaf, fill):
        return jnp.concatenate([leaf, jnp.full((n_off, 4), fill, leaf.dtype)], axis=1)

    tail_padded = SimEvent(type=append(exp_.events.type, EventType.DELAY),
                           dt=append(exp_.events.dt, 0.0),
                           rf_amp=append(exp_.events.rf_amp, 0.0),
                           rf_phase=append(exp_.events.rf_phase, 0.0),
                           rf_freq=append(exp_.events.rf_freq, 0.0),
                           adc_idx=append(exp_.events.adc_idx, -1))
    z_wrong = np.asarray(simulate_spectrum(exp_.params, tail_padded, exp_.m0,
                                           return_full_trajectory=False)[:, exp_.mz_loc])

    assert np.ptp(z_correct) > 0.1, "the correct spectrum should show real saturation"
    assert np.ptp(z_wrong) < 1e-6, "appending should have flattened it -- the trap is gone?"
    assert np.abs(z_wrong - exp_.m0[exp_.mz_loc]).max() < 1e-6, (
        "appended padding returns the STARTING magnetisation, which is the whole point")


def test_stacking_equal_length_sequences():
    """A QUESP ladder: same saturation time, different powers, so the widths already match."""
    parts = [experiment_at(2.0, b1_uT=b) for b in (0.5, 1.0, 2.0)]
    joint = Experiment.stack(parts)
    assert joint.n_offsets == 3 * PPM.size
    assert joint.n_events == parts[0].n_events

    z = np.asarray(joint.simulate())
    for i, part in enumerate(parts):
        block = z[i * PPM.size:(i + 1) * PPM.size]
        assert np.abs(block - np.asarray(part.simulate())).max() == 0.0


def test_stacking_sequences_of_different_lengths():
    """The QUEST case: sequences that differ in event count stack once they are padded."""
    parts = [experiment_at(0.5), experiment_at(2.0), experiment_at(4.0)]
    # make them genuinely different lengths, as a real pulsed QUEST ladder is
    parts = [Experiment(p.params, pad_events(p.events, p.n_events + 2 * i), p.m0, p.mz_loc)
             for i, p in enumerate(parts)]
    widths = {p.n_events for p in parts}
    assert len(widths) == 3, "the test needs genuinely ragged inputs"

    joint = Experiment.stack(parts)
    assert joint.n_events == max(widths)
    assert joint.n_offsets == 3 * PPM.size

    z = np.asarray(joint.simulate())
    for i, part in enumerate(parts):
        block = z[i * PPM.size:(i + 1) * PPM.size]
        assert np.abs(block - np.asarray(part.simulate())).max() == 0.0, (
            f"block {i} changed when stacked; padding is not a no-op")


def test_stack_rejects_mismatched_models():
    a = experiment_at(2.0, name="amide")
    b = experiment_at(2.0, name="amide_noe")          # 2 CEST pools -> different mz_loc
    assert a.mz_loc != b.mz_loc
    with pytest.raises(ValueError, match="mz_loc"):
        Experiment.stack([a, b])


def test_stack_rejects_nothing_to_stack():
    with pytest.raises(ValueError, match="at least one"):
        Experiment.stack([])


def test_pad_cannot_shrink():
    exp_ = experiment_at(2.0)
    with pytest.raises(ValueError, match="cannot pad"):
        pad_events(exp_.events, exp_.n_events - 1)
