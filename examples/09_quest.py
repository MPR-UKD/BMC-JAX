"""Tutorial 09: QUEST fit of the same phantom

The seven tubes of tutorial 08 were measured in the same session with QUEST: six saturation times
from 0.4 to 5 s (2 to 25 pulses) at an amplitude of 1.0 uT. QUESP and QUEST determine the exchange
rate through different mechanisms (tutorial 06), so comparing them is a meaningful cross-check.
This tutorial fits the same three parameters per tube with the same fixed inputs and the same
bounds as tutorial 08. Only the experiment differs, so the two results can be compared directly.

This tutorial
  1. loads the six QUEST rungs. A longer saturation contains more pulses, so the rungs have
     different numbers of events, and Experiment.stack() pads them to the same length;
  2. fits k, f and the pool T2 of every tube;
  3. prints the result and the pair checks.

Comparison with tutorial 08: where QUESP and QUEST agree, the result is supported by two different
mechanisms. Where they disagree, at least one experiment does not determine the parameter. For slow
exchange, a single amplitude (QUEST) determines mainly the product k*f, not k and f separately, so
the slow tubes are expected to differ.

Requirements: the core package; the data ship in examples/data/.

Run:  python 09_quest.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phantom import (EXCLUDE_PPM, POOL_T2_START, load_data, load_events, print_pairs, quest_stem,
                      start_magnetisation, vial_params)

from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit

print(__doc__)

data = load_data(tubes="pool")
ppm, tsat, names = data["ppm"], data["tsat_s"], [str(n) for n in data["vial_names"]]
keep = np.abs(ppm) >= EXCLUDE_PPM

# ---------------------------------------------------------------------------------------------
# 1. The sequences, and rungs of different length
# ---------------------------------------------------------------------------------------------
# Experiment.stack pads every rung to the length of the longest with zero-duration events placed
# before the sequence, so that one compiled loop runs all rungs. The padding is never placed after
# the readout: the simulator returns the last magnetisation it recorded and an ADC event resets the
# running state, so an event after the readout would return the start magnetisation. The padding
# has no duration either, because that would add free relaxation to the shortest rung, which
# carries the most information on the exchange rate.

print(f"Ladder: Tsat = {', '.join(f'{t:g}' for t in tsat)} s (at 1.0 uT)")

params = vial_params(data)
m0 = start_magnetisation(params)
rungs = [Experiment(params, load_events(quest_stem(t), keep), m0, 4) for t in tsat]
print("  events per rung: " + ", ".join(f"{r.n_events}" for r in rungs))

ladder = Experiment.stack(rungs)
Z = data["Z_quest"][:, :, keep].reshape(len(names), -1)
print(f"  padded to {ladder.n_events} each, {ladder.n_offsets} offsets in total")
print(f"Model: {ladder.n_offsets} offsets    Data: {Z.shape[0]} tubes x {Z.shape[1]} points")

# ---------------------------------------------------------------------------------------------
# 2. The fit
# ---------------------------------------------------------------------------------------------
# Identical to tutorial 08: the same three free parameters, the same fixed per-tube constants, the
# same bounds. Only the ladder differs, which is what makes the comparison meaningful.

spec = FitSpec([FreeParam("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=2000.0, x0=300.0),
                FreeParam("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008),
                FreeParam("r2p", "cest_r2", 0, scale=100.0, lo=10.0, hi=1000.0, x0=1.0 / POOL_T2_START)])
res = fit(ladder, spec, Z, n_iter=60, n_per_block=int(keep.sum()))

# ---------------------------------------------------------------------------------------------
# 3. The result
# ---------------------------------------------------------------------------------------------

noise = data["noise_quest"]
print(f"\n{'tube':<18} {'k [/s]':>8} {'f':>9} {'T2 pool':>8} {'rms/noise':>10}  on a bound")
print("-" * 70)
for i, name in enumerate(names):
    bound = ", ".join(s.name for s, b in zip(spec.slots, res.at_bounds[i]) if b) or "-"
    print(f"{name:<18} {res['k'][i]:>8.1f} {res['f'][i]:>9.5f} {1e3 / res['r2p'][i]:>6.1f}ms "
          f"{res.rms[i] / noise[i]:>10.1f}  {bound}")
print()
print_pairs(res["f"], names)

print("""
Compare with the output of 08_quesp.py. Where the two ladders agree on a tube, the result is
supported by two different mechanisms. Where they disagree, at least one of them does not measure
the exchange rate. A single fit cannot show this on its own, which is the reason to measure both.
The slow tubes are expected to differ: at a single saturation amplitude, QUEST determines mainly
the product k*f of a slow pool, not k and f separately.

Next: 10_shared_water.py, a joint fit with the water parameters shared by all tubes.""")
