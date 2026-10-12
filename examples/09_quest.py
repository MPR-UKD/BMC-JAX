"""09 -- QUEST on the same phantom: six tubes, six saturation times.

QUESP sweeps how HARD you saturate; QUEST sweeps how LONG. They reach the exchange rate through
different physics -- QUESP through the saturation efficiency, QUEST through how fast the label
builds up in water -- so the two are a genuine cross-check and not a repeat measurement. These
are the same tubes as tutorial 08, scanned in the same session.

QUEST also brings a mechanical problem QUESP does not have: the rungs are not the same length.
A longer saturation is more pulses, so the sequences have different event counts and cannot be
batched as they are.

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
# 1. The sequences, and the ragged-rung problem
# ---------------------------------------------------------------------------------------------
# Experiment.stack pads every rung to the longest with leading no-ops, so one compiled loop runs
# them all. The padding goes BEFORE the sequence, never after: the simulator returns the last
# magnetisation it saw and an ADC resets the running state, so a no-op after the readout would
# return the STARTING magnetisation -- a smooth, flat, entirely wrong spectrum. The no-ops carry
# no duration either, since that would be free relaxation added to precisely the shortest rung,
# the one carrying the rate information.

print(f"Ladder: Tsat = {', '.join(f'{t:g}' for t in tsat)} s (at 1.0 uT)")

params = vial_params(data)
m0 = start_magnetisation(params)
rungs = [Experiment(params, load_events(quest_stem(t), keep), m0, 4) for t in tsat]
print("  events per rung: " + ", ".join(f"{r.n_events}" for r in rungs))

ladder = Experiment.stack(rungs)
Z = data["Z_quest"][:, :, keep].reshape(len(names), -1)
print(f"  padded to {ladder.n_events} each -> {ladder.n_offsets} offsets")
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
Run 08_quesp.py and compare. Where the two ladders agree on a tube, the number rests on two
different pieces of physics and is worth believing. Where they disagree, at least one of them
is reporting something other than the exchange rate -- and the fit will not tell you that on
its own, which is the reason to measure both. Expect the slow tubes to disagree: at a single
saturation power QUEST fixes mainly the product k*f of a slow pool, not k and f apart.""")
