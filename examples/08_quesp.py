"""08 -- QUESP on real phantom data: four vials, seven saturation powers.

Tutorials 04 to 06 fit simulations, where the answer is known because it was put in. This one
fits a measurement. Four tubes of known concentration were scanned at 2.8936 T with the seven-rung
B1 ladder, and the exchange rate and pool size of each are solved jointly across the whole ladder.

Everything is shipped: the Z-spectra, the sequences that produced them, and the per-vial
constants the fit holds fixed. Nothing outside this package is needed.

Run:  python 08_quesp.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phantom import (EXCLUDE_PPM, load_data, load_events, start_magnetisation, vial_params)

from bmc_jax.fitting import Experiment, FitSpec, Slot, fit

print(__doc__)

data = load_data()
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
ladder = Experiment.stack([
    Experiment(params, load_events(f"quesp_sim_b1_{('%g' % b).replace('.', 'p')}uT", keep), m0, 4)
    for b in levels])
Z = data["Z_quesp"][:, :, keep].reshape(len(names), -1)
print(f"Model: {ladder.n_offsets} offsets    Data: {Z.shape[0]} vials x {Z.shape[1]} points")

# ---------------------------------------------------------------------------------------------
# 2. The fit
# ---------------------------------------------------------------------------------------------
# Two free parameters per vial. Everything else -- water R1 and R2, B0, relative B1, the pool's
# chemical shift and T2, the starting magnetisation -- was measured separately and is held
# fixed. `scale` is the size of a natural step in each parameter, not a bound.

spec = FitSpec([Slot("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=5000.0, x0=300.0),
                Slot("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008)])
res = fit(ladder, spec, Z, n_iter=20, n_per_block=int(keep.sum()))

# ---------------------------------------------------------------------------------------------
# 3. The result
# ---------------------------------------------------------------------------------------------
# f_expected is what the concentration weighed into each tube implies, so f / f_expected is the
# fit against an independent measurement of the same quantity -- 1.0 would be exact agreement.

print(f"\n{'vial':<18} {'k [Hz]':>8} {'f':>9} {'f / f_exp':>10} {'RMS':>9}")
print("-" * 58)
for i, name in enumerate(names):
    print(f"{name:<18} {res['k'][i]:>8.1f} {res['f'][i]:>9.5f} "
          f"{res['f'][i] / data['f_expected'][i]:>10.3f} {res.rms_per_block[i].mean():>9.5f}")

# The two creatine tubes are the same solution at the same concentration, so their fitted pool
# sizes must agree whatever the absolute calibration is. No systematic error shared by both can
# satisfy this, which makes it worth more than any single absolute number above.
ratio = res["f"][names.index("creatine_pH6.25")] / res["f"][names.index("creatine_pH7.87")]
print(f"\ncreatine pair f-ratio: {ratio:.3f}   (must be 1.00 -- same solution, same concentration)")
print(f"at_bounds: {res.at_bounds}   (a parameter on its bound is not a measurement)")

print("\nNext: 09_quest.py -- the same four vials measured the other way, in saturation TIME.")
