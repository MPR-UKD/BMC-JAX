"""Tutorial 06: The QUEST ladder (several saturation times)

QUEST (quantification of exchange as a function of saturation time) measures the same sample with
several saturation times at one amplitude. QUESP and QUEST determine the exchange rate through
different mechanisms: QUESP through the saturation efficiency, QUEST through the build-up of the
saturation in water over time. Agreement between the two is therefore a meaningful cross-check
and not a repeated measurement. In practice, QUEST also uses one moderate amplitude, which avoids
the power limits that restrict the high amplitudes of a QUESP ladder.

This tutorial
  1. builds a QUEST ladder (Tsat = 0.2 to 4 s at 1 uT). A longer saturation has more events, so the
     rungs differ in length. Experiment.stack() pads the shorter rungs so that all rungs have the
     same shape and run in one compiled loop;
  2. fits k and f with QUESP and with QUEST at three exchange rates and compares the two results.

How the padding works: the added events have zero duration and are placed before the sequence.
Placed after the readout, they would make the simulator return the start magnetisation instead of
the measured one. With a nonzero duration, they would add free relaxation to the shortest rungs,
which carry most of the information on the exchange rate.

Requirements: the core package only.

Run:  python 06_quest.py
"""
import numpy as np

from bmc_jax.fitting import FitSpec, Experiment, FreeParam, fit
from bmc_jax.simulation import pad_events
from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state
from bmc_jax.sequences import cw_saturation

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
m0 = initial_state(jax_params)
ppm = np.linspace(-6.0, 6.0, 41)
TSAT = (0.2, 0.5, 1.0, 2.0, 4.0)
NOISE = 0.002

# ---------------------------------------------------------------------------------------------
# Rungs of different length, and how stacking handles them
# ---------------------------------------------------------------------------------------------
# The rungs of a QUESP ladder all have the same number of events: only the amplitude changes. The
# rungs of a QUEST ladder do not, because a longer saturation contains more pulses. Stacking pads
# the shorter rungs so that every rung has the same shape, which lets one compiled loop run them
# all.
#
# The padding consists of zero-duration events placed before the sequence. Not after it: the
# simulator returns the last magnetisation it recorded, and an ADC event resets the running state,
# so an event after the readout would return the start magnetisation, a flat and wrong spectrum.
# Not with a duration either: that would be free relaxation, added to the shortest rung, which
# carries the most information on the exchange rate.

rungs = [Experiment(jax_params, cw_saturation(ppm, b1_uT=1.0, tsat_s=t, b0_T=params.system.b0),
                 m0, params.mz_loc) for t in TSAT]
# give the rungs different lengths, as in a pulsed saturation train
rungs = [Experiment(r.params, pad_events(r.events, r.n_events + 2 * i), r.m0, r.mz_loc)
         for i, r in enumerate(rungs)]
print(f"events per rung before stacking: {[r.n_events for r in rungs]}")

quest = Experiment.stack(rungs)
print(f"after stacking: {quest.n_offsets} offsets x {quest.n_events} events "
      f"(padded to the longest)\n")

quesp = Experiment.stack([
    Experiment(jax_params, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0), m0,
            params.mz_loc) for b in (0.25, 0.5, 1.0, 2.0, 4.0)])


def spec(x0_k):
    return FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=x0_k),
                    FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0003)])


def measure(problem, k_true, seed=0):
    truth = jax_params._replace(cest_k=np.float32([k_true]))
    z = np.asarray(Experiment(truth, problem.events, initial_state(truth),
                           params.mz_loc).simulate())
    return (z + np.random.default_rng(seed).normal(0, NOISE, z.shape))[None, :]


# ---------------------------------------------------------------------------------------------
# QUESP against QUEST
# ---------------------------------------------------------------------------------------------
print(f"Tsat ladder {TSAT} s at 1.0 uT, against a B1 ladder at 2.0 s:")
print(f"{'k true':>8} | {'QUESP':>8} {'err':>6} | {'QUEST':>8} {'err':>6} | {'they agree to':>14}")
print("-" * 68)
for k_true in (150.0, 500.0, 1500.0):
    out = []
    for prob in (quesp, quest):
        blank = Experiment(jax_params, prob.events, m0, params.mz_loc)
        k = fit(blank, spec(3 * k_true), measure(prob, k_true), n_iter=80)["k"][0]
        out += [k, abs(k / k_true - 1)]
    agree = abs(out[0] / out[2] - 1)
    print(f"{k_true:>8.0f} | {out[0]:>8.0f} {out[1]:>5.0%} | {out[2]:>8.0f} {out[3]:>5.0%} "
          f"| {agree:>13.0%}")

print("""
The agreement between the two experiments is the main result. Both use the same model but a
different mechanism, so a relevant error of the model makes them disagree. When they agree, the
exchange rate is a property of the sample and not of one experiment.

The two are not fully independent: both use the same pool T2, the same relaxation times and the
same B0 and B1 calibration. A disagreement points to the model; an agreement does not prove that
these fixed inputs are correct.

Next: 07_many_voxels.py, the same fit applied to many voxels at once.""")
