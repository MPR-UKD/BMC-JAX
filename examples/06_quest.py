"""06 -- The other ladder: varying saturation TIME instead of power (QUEST).

QUESP sweeps how hard you saturate. QUEST sweeps how long. They reach the exchange rate through
different physics -- QUESP through the saturation efficiency, QUEST through how fast the label
builds up in water -- so agreement between them is a real check and not a repeat measurement.

QUEST also matters practically: it runs at one fixed, modest B1, so it does not hit the power
limits that stop a QUESP ladder from reaching the high amplitudes a fast pool needs.

Run:  python 06_quest.py
"""
import numpy as np

from bmc_jax.fitting import FitSpec, Problem, Slot, fit, pad_events
from bmc_jax.io import example_config
from bmc_jax.jax_solver import convert_params, cw_saturation, get_initial_magnetization

print(__doc__)

params = example_config("amide")
jax_params = convert_params(params)
m0 = get_initial_magnetization(jax_params)
ppm = np.linspace(-6.0, 6.0, 41)
TSAT = (0.2, 0.5, 1.0, 2.0, 4.0)
NOISE = 0.002

# ---------------------------------------------------------------------------------------------
# The ragged-sequence problem, and how stacking solves it
# ---------------------------------------------------------------------------------------------
# A QUESP ladder's rungs all have the same number of events -- only the amplitude changes. A
# QUEST ladder's do not: a longer saturation is more pulses. Stacking pads the short ones so
# every rung has the same shape, which is what lets one compiled loop run them all.
#
# The padding is zero-duration no-ops placed BEFORE the sequence. Not after: the simulator
# returns the last magnetisation it saw, and an ADC resets the running state, so a no-op after
# the readout would return the STARTING magnetisation -- a smooth, flat, entirely wrong
# spectrum. Not with a duration either: that is free relaxation, and it would add recovery time
# to exactly the shortest rung, the one carrying the rate information.

rungs = [Problem(jax_params, cw_saturation(ppm, b1_uT=1.0, tsat_s=t, b0_T=params.system.b0),
                 m0, params.mz_loc) for t in TSAT]
# make them genuinely ragged, as a pulsed train would be
rungs = [Problem(r.params, pad_events(r.events, r.n_events + 2 * i), r.m0, r.mz_loc)
         for i, r in enumerate(rungs)]
print(f"events per rung before stacking: {[r.n_events for r in rungs]}")

quest = Problem.stack(rungs)
print(f"after stacking: {quest.n_offsets} offsets x {quest.n_events} events "
      f"(padded to the longest)\n")

quesp = Problem.stack([
    Problem(jax_params, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0), m0,
            params.mz_loc) for b in (0.25, 0.5, 1.0, 2.0, 4.0)])


def spec(x0_k):
    return FitSpec([Slot("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=x0_k),
                    Slot("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0003)])


def measure(problem, k_true, seed=0):
    truth = jax_params._replace(cest_k=np.float32([k_true]))
    z = np.asarray(Problem(truth, problem.events, get_initial_magnetization(truth),
                           params.mz_loc).simulate())
    return (z + np.random.default_rng(seed).normal(0, NOISE, z.shape))[None, :]


# ---------------------------------------------------------------------------------------------
# Two independent routes to the same number
# ---------------------------------------------------------------------------------------------
print(f"Tsat ladder {TSAT} s at 1.0 uT, against a B1 ladder at 2.0 s:")
print(f"{'k true':>8} | {'QUESP':>8} {'err':>6} | {'QUEST':>8} {'err':>6} | {'they agree to':>14}")
print("-" * 68)
for k_true in (150.0, 500.0, 1500.0):
    out = []
    for prob in (quesp, quest):
        blank = Problem(jax_params, prob.events, m0, params.mz_loc)
        k = fit(blank, spec(3 * k_true), measure(prob, k_true), n_iter=80)["k"][0]
        out += [k, abs(k / k_true - 1)]
    agree = abs(out[0] / out[2] - 1)
    print(f"{k_true:>8.0f} | {out[0]:>8.0f} {out[1]:>5.0%} | {out[2]:>8.0f} {out[3]:>5.0%} "
          f"| {agree:>13.0%}")

print("""
Agreement between the two is the useful result. They share the model but not the mechanism, so
if the model is wrong in a way that matters they drift apart -- and if they agree, the number is
a property of the sample rather than of one experiment.

They are not fully independent: both use the same pool T2, the same relaxation times and the
same B0/B1 calibration. A disagreement points at the model; an agreement does not prove those
fixed inputs are right.

Next: 07_many_voxels.py -- the same fit over an image instead of one spectrum.""")
