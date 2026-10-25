"""Tutorial 08: QUESP fit of a measured phantom

Tutorials 04 to 07 fitted simulated data, where the true values are known. This tutorial fits a
measurement: a phantom of seven tubes, measured on 2026-09-27 on a 3 T scanner (2.8936 T). Tube 0
contains phosphate-buffered saline (PBS) without an exchanging agent. The other six tubes form three
pairs: the glycosaminoglycan (GAG) chondroitin sulfate at 100 and 50 mg/mL, creatine at pH 7.51
and 6.31, and carnosine at pH 6.59 and 7.44. QUESP was measured with eight saturation amplitudes
from 0.15 to 1.8 uT.

This tutorial
  1. loads the Z-spectra averaged over each tube, and the scanner sequences converted to events;
  2. excludes the offsets within 1 ppm of water, where direct water saturation dominates;
  3. stacks the eight amplitudes into one experiment and fits three parameters per tube: the
     exchange rate k, the pool size f and the T2 of the CEST pool (fitted as its rate R2);
  4. prints the result, the misfit relative to the noise (rms/noise) and the pair checks.

Held fixed: the water R1 of each tube, the water R2 (from the PBS tube), the B0 offset and the
relative B1 of each tube (from WASABI maps), and the start magnetisation M_start.

Pair checks: the absolute pool size depends on quantities the fit cannot see, such as purity, the
number of labile protons and the molar mass. The ratio within a pair does not: the GAG pair must
give an f ratio of 2, and the creatine and carnosine pairs, each one solution at two pH values, a
ratio of 1.

Requirements: the core package; the data ship in examples/data/. Most of the runtime is the
compilation of the long scanner sequences.

Run:  python 08_quesp.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phantom import (EXCLUDE_PPM, M_START, POOL_T2_START, load_data, load_events, print_pairs, quesp_stem,
                      start_magnetisation, vial_params)

from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit

print(__doc__)

data = load_data(tubes="pool")          # the six tubes with a CEST pool; PBS has nothing to fit here
ppm, levels, names = data["ppm"], data["b1_levels_uT"], [str(n) for n in data["vial_names"]]
keep = np.abs(ppm) >= EXCLUDE_PPM

# ---------------------------------------------------------------------------------------------
# 1. The sequences
# ---------------------------------------------------------------------------------------------
# One converted sequence per rung, joined along the offset axis so that one residual spans the
# whole ladder. This joint residual makes k and f separable: no rung has to be well conditioned
# on its own, because the change across the rungs carries the information (tutorial 05).
#
# The measurement must be concatenated in the same order of rungs. A wrong order raises no error;
# it only gives a wrong fit.

print(f"Ladder: B1 = {', '.join(f'{b:g}' for b in levels)} uT")
print(f"Offsets: {ppm.size} measured, {int(keep.sum())} kept (|ppm| >= {EXCLUDE_PPM})")

params = vial_params(data)
m0 = start_magnetisation(params)
ladder = Experiment.stack([Experiment(params, load_events(quesp_stem(b), keep), m0, 4) for b in levels])
Z = data["Z_quesp"][:, :, keep].reshape(len(names), -1)
print(f"Model: {ladder.n_offsets} offsets    Data: {Z.shape[0]} tubes x {Z.shape[1]} points")

# ---------------------------------------------------------------------------------------------
# 2. The fit
# ---------------------------------------------------------------------------------------------
# Three free parameters per tube: the exchange rate, the pool size and the T2 of the pool (fitted
# as its rate R2), which was not measured for this phantom. All properties of the water (R1, R2,
# B0, relative B1 and the start magnetisation M_start, fitted once to the PBS tube alone) were
# determined separately and are held fixed. `scale` is the typical size of each parameter, not a
# bound.

spec = FitSpec([FreeParam("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=2000.0, x0=300.0),
                FreeParam("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008),
                FreeParam("r2p", "cest_r2", 0, scale=100.0, lo=10.0, hi=1000.0, x0=1.0 / POOL_T2_START)])
print(f"Water held: M_start {M_START}, water R2 {float(data['water_r2']):.3f} /s (PBS) for every tube")
res = fit(ladder, spec, Z, n_iter=60, n_per_block=int(keep.sum()))

# ---------------------------------------------------------------------------------------------
# 3. The result
# ---------------------------------------------------------------------------------------------
# rms/noise compares the misfit with the noise of the tube's mean spectrum, measured far from water:
# a value of 1 would be a fit to the noise level. A parameter on its bound is not a measurement:
# the optimiser was stopped at the bound.

noise = data["noise_quesp"]
print(f"\n{'tube':<18} {'k [/s]':>8} {'f':>9} {'T2 pool':>8} {'rms/noise':>10}  on a bound")
print("-" * 70)
for i, name in enumerate(names):
    bound = ", ".join(s.name for s, b in zip(spec.slots, res.at_bounds[i]) if b) or "-"
    print(f"{name:<18} {res['k'][i]:>8.1f} {res['f'][i]:>9.5f} {1e3 / res['r2p'][i]:>6.1f}ms "
          f"{res.rms[i] / noise[i]:>10.1f}  {bound}")

# Absolute pool sizes depend on quantities the fit cannot see (purity, labile protons per molecule,
# molar mass). A ratio within a pair does not: GAG at 100 and 50 mg/mL must give 2, and the same
# creatine or carnosine solution at two pH values must give 1. These ratios are therefore stronger
# checks than the absolute values above.
print()
print_pairs(res["f"], names)
print("\nNote: the carnosine pool at pH 6.59 is small and exchanges slowly. Its CEST effect is close to the"
      "\nnoise at these amplitudes, so its fitted values are uncertain.")

print("\nNext: 09_quest.py, the same tubes measured with varying saturation time.")
