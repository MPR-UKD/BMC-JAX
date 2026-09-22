"""Stacking several sequences into one fit, and the padding that makes it possible.

The load-bearing test here is `test_padding_at_the_end_would_destroy_the_answer`. It builds the
wrong padding deliberately and shows what it does, so the reason the code pads at the front is
recorded as evidence rather than as a claim in a comment.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmc_jax.fitting import Problem, pad_events
from bmc_jax.io import example_config
from bmc_jax.jax_solver import (EventType, SimEvent, convert_params, cw_saturation,
                                get_initial_magnetization, simulate_spectrum)

PPM = np.linspace(-5.0, 5.0, 21)


def problem_at(tsat_s, b1_uT=1.0, name="amide"):
    """One CW saturation experiment. Longer tsat does NOT mean more events here -- see below."""
    p = example_config(name)
    jp = convert_params(p)
    ev = cw_saturation(PPM, b1_uT=b1_uT, tsat_s=tsat_s, b0_T=p.system.b0)
    return Problem(jp, ev, get_initial_magnetization(jp), p.mz_loc)


def test_front_padding_changes_nothing():
    """Zero-duration DELAYs before the sequence are a true no-op, to the last bit."""
    prob = problem_at(2.0)
    z_plain = np.asarray(prob.simulate())
    padded = Problem(prob.params, pad_events(prob.events, prob.n_events + 37),
                     prob.m0, prob.mz_loc)
    assert padded.n_events == prob.n_events + 37
    assert np.abs(z_plain - np.asarray(padded.simulate())).max() == 0.0


def test_padding_at_the_end_would_destroy_the_answer():
    """Why the padding goes at the front, demonstrated rather than asserted.

    `simulate_sequence` returns `final_state.last_M`, which every step overwrites, and an ADC
    with `reset_after_adc` sets the running magnetisation back to its initial value. A no-op
    appended AFTER the readout therefore copies that reset value into `last_M`, and the function
    returns the STARTING magnetisation instead of the saturated one.

    The failure is silent: the result is a smooth, plausible, completely flat Z-spectrum.
    """
    prob = problem_at(2.0)
    z_correct = np.asarray(prob.simulate())

    n_off, n_ev = prob.events.dt.shape

    def append(leaf, fill):
        return jnp.concatenate([leaf, jnp.full((n_off, 4), fill, leaf.dtype)], axis=1)

    tail_padded = SimEvent(type=append(prob.events.type, EventType.DELAY),
                           dt=append(prob.events.dt, 0.0),
                           rf_amp=append(prob.events.rf_amp, 0.0),
                           rf_phase=append(prob.events.rf_phase, 0.0),
                           rf_freq=append(prob.events.rf_freq, 0.0),
                           adc_idx=append(prob.events.adc_idx, -1))
    z_wrong = np.asarray(simulate_spectrum(prob.params, tail_padded, prob.m0,
                                           return_full_trajectory=False)[:, prob.mz_loc])

    assert np.ptp(z_correct) > 0.1, "the correct spectrum should show real saturation"
    assert np.ptp(z_wrong) < 1e-6, "appending should have flattened it -- the trap is gone?"
    assert np.abs(z_wrong - prob.m0[prob.mz_loc]).max() < 1e-6, (
        "appended padding returns the STARTING magnetisation, which is the whole point")


def test_stacking_equal_length_sequences():
    """A QUESP ladder: same saturation time, different powers, so the widths already match."""
    parts = [problem_at(2.0, b1_uT=b) for b in (0.5, 1.0, 2.0)]
    joint = Problem.stack(parts)
    assert joint.n_offsets == 3 * PPM.size
    assert joint.n_events == parts[0].n_events

    z = np.asarray(joint.simulate())
    for i, part in enumerate(parts):
        block = z[i * PPM.size:(i + 1) * PPM.size]
        assert np.abs(block - np.asarray(part.simulate())).max() == 0.0


def test_stacking_sequences_of_different_lengths():
    """The QUEST case: sequences that differ in event count stack once they are padded."""
    parts = [problem_at(0.5), problem_at(2.0), problem_at(4.0)]
    # make them genuinely different lengths, as a real pulsed QUEST ladder is
    parts = [Problem(p.params, pad_events(p.events, p.n_events + 2 * i), p.m0, p.mz_loc)
             for i, p in enumerate(parts)]
    widths = {p.n_events for p in parts}
    assert len(widths) == 3, "the test needs genuinely ragged inputs"

    joint = Problem.stack(parts)
    assert joint.n_events == max(widths)
    assert joint.n_offsets == 3 * PPM.size

    z = np.asarray(joint.simulate())
    for i, part in enumerate(parts):
        block = z[i * PPM.size:(i + 1) * PPM.size]
        assert np.abs(block - np.asarray(part.simulate())).max() == 0.0, (
            f"block {i} changed when stacked; padding is not a no-op")


def test_stack_rejects_mismatched_models():
    a = problem_at(2.0, name="amide")
    b = problem_at(2.0, name="amide_noe")          # 2 CEST pools -> different mz_loc
    assert a.mz_loc != b.mz_loc
    with pytest.raises(ValueError, match="mz_loc"):
        Problem.stack([a, b])


def test_stack_rejects_nothing_to_stack():
    with pytest.raises(ValueError, match="at least one"):
        Problem.stack([])


def test_pad_cannot_shrink():
    prob = problem_at(2.0)
    with pytest.raises(ValueError, match="cannot pad"):
        pad_events(prob.events, prob.n_events - 1)
