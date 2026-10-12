"""08 -- QUESP on real phantom data: six tubes, eight saturation powers.

Tutorials 04 to 06 fit simulations, where the answer is known because it was put in. This one
fits a measurement. A seven-tube phantom (PBS plus GAG, creatine and carnosine pairs) was scanned
at 2.8936 T with an eight-rung B1 ladder, and the exchange rate, pool size and pool T2 of each tube
with a CEST pool are solved jointly across the whole ladder.

Everything is shipped: the Z-spectra, the sequences that produced them, and the per-tube
constants the fit holds fixed. Nothing outside this package is needed.

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
# One converted sequence per rung, joined along the offset axis so a single residual spans the
# whole ladder. That joint residual is what makes k and f separable: no rung has to be well
# conditioned on its own, because the shape ACROSS the rungs carries the information.
#
# The measurement must be concatenated in the same rung order. Getting that wrong does not
# raise -- it just produces a fit.

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
# Three free parameters per tube: exchange rate, pool size and the pool's T2 (as its R2), which was
# not measured for this phantom. Everything about the water -- R1 and R2, B0, relative B1, the
# starting magnetisation (M_start, fitted once to the PBS tube alone) -- was measured separately and
# is held fixed. `scale` is the size of a natural step in each parameter, not a bound.

spec = FitSpec([FreeParam("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=2000.0, x0=300.0),
                FreeParam("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008),
                FreeParam("r2p", "cest_r2", 0, scale=100.0, lo=10.0, hi=1000.0, x0=1.0 / POOL_T2_START)])
print(f"Water held: M_start {M_START}, water R2 {float(data['water_r2']):.3f} /s (PBS) for every tube")
res = fit(ladder, spec, Z, n_iter=60, n_per_block=int(keep.sum()))

# ---------------------------------------------------------------------------------------------
# 3. The result
# ---------------------------------------------------------------------------------------------
# rms / noise compares the misfit with the tube-mean noise measured on the far wing: 1 would be a fit
# to the noise. A parameter on its bound is not a measurement -- the optimiser wanted to go further.

noise = data["noise_quesp"]
print(f"\n{'tube':<18} {'k [/s]':>8} {'f':>9} {'T2 pool':>8} {'rms/noise':>10}  on a bound")
print("-" * 70)
for i, name in enumerate(names):
    bound = ", ".join(s.name for s, b in zip(spec.slots, res.at_bounds[i]) if b) or "-"
    print(f"{name:<18} {res['k'][i]:>8.1f} {res['f'][i]:>9.5f} {1e3 / res['r2p'][i]:>6.1f}ms "
          f"{res.rms[i] / noise[i]:>10.1f}  {bound}")

# Absolute pool sizes depend on things no fit sees (purity, labile protons per molecule, the
# concentration on the label). A ratio within a pair does not: GAG at 100 and 50 mg/mL must give 2,
# and the same creatine or carnosine solution at two pH must give 1. Worth more than any single
# absolute number above.
print()
print_pairs(res["f"], names)
print("\nThe slow, small carnosine pool at pH 6.59 is barely visible at these powers: read its numbers with care.")

print("\nNext: 09_quest.py -- the same tubes measured the other way, in saturation TIME.")
