"""11 -- One value per rung of the ladder, shared by the tubes.

Some errors belong to a rung, not to a tube. The scanner plays each rung of the QUESP ladder at its
own RF power; if the amplifier is not perfectly linear, the B1 a rung delivers can differ from the
nominal by a few percent -- the same for every tube, and invisible to a B1 map measured at one power.
A slow field drift does the same to B0: one offset per rung, shared by the tubes.

`FreeParam(scope="block")` is one value per experiment of a stacked ladder, shared by the samples:

  B1 per rung   field "rel_b1", mode "factor": the measured relative B1 of every tube times the rung's factor
  (B0 per rung  field "b0_inhom", mode "add" would be the drift; not fitted here, the data are B0-corrected)

This runs tutorial 10's model (k, f and pool T2 per pool tube; water R2 factor, M_start and B0 spread
shared by all seven tubes), then the same with a B1 factor per rung, and compares them rung by rung.
`block_jacobian="split"` builds the eight B1 columns from ONE forward tangent instead of eight (the same
derivative; see fit_joint's notes).

What to look for:
  * factors near 1 with no pattern: the amplifier is linear and the per-rung B1 explains nothing;
  * a smooth trend with power: an amplifier effect, worth a B1 map at a second power;
  * the mean of the factors trades against the water R2 factor (both deepen the water saturation), so
    read them together;
  * the PBS tube: pure water, no pool. A real B1 error changes its saturation too, so a real per-rung B1
    must fit PBS better at the rungs it changes. If only the pool tubes improve, the factors are soaking
    up a pool-model error, not measuring B1.

Run:  python 11_per_rung.py          (about 3 h on one RTX 3070: two joint fits with 15 nodes)
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phantom import (EXCLUDE_PPM, M_START, POOL_T2_START, load_data, load_events, print_pairs, quesp_stem,
                      start_magnetisation, vial_params)

from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit, fit_joint
from bmc_jax.simulation import B0Spread

print(__doc__)

data = load_data()
ppm, levels, names = data["ppm"], data["b1_levels_uT"], [str(n) for n in data["vial_names"]]
pool = tuple(int(i) for i in np.flatnonzero(np.isfinite(data["vial_dw"])))
keep = np.abs(ppm) >= EXCLUDE_PPM
n_per = int(keep.sum())


def quesp(d):
    """The QUESP ladder and its data for the tubes in `d`."""
    p = vial_params(d)
    lad = Experiment.stack([Experiment(p, load_events(quesp_stem(b), keep), start_magnetisation(p), 4)
                            for b in levels])
    return lad, d["Z_quesp"][:, :, keep].reshape(len(d["vial_names"]), -1)


K = FreeParam("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=2000.0, x0=300.0)
F = FreeParam("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008)
T2 = FreeParam("r2p", "cest_r2", 0, scale=100.0, lo=10.0, hi=1000.0, x0=1.0 / POOL_T2_START)
R2F = FreeParam("r2w", "water_r2", None, scale=1.0, lo=0.5, hi=2.0, x0=1.0, scope="shared", mode="factor")
MS = FreeParam("m_start", "m_start", None, scale=1.0, lo=0.2, hi=1.0, x0=M_START, scope="shared")
SIG = FreeParam("sigma", "b0_spread", None, scale=10.0, lo=0.0, hi=8.0, x0=5.0, scope="shared")
B1R = FreeParam("b1", "rel_b1", None, scale=1.0, lo=0.7, hi=1.3, x0=1.0, scope="block", mode="factor")
spread = B0Spread(nodes=15)
progress = lambda it, c: print(f"   step {it + 1:2d}  cost {c:.5g}", flush=True)

print("1. k, f and pool T2 per tube (tutorial 08), for the starts ...")
ladder6, Z6 = quesp(load_data(tubes=pool))
plain = fit(ladder6, FitSpec([K, F, T2]), Z6, n_iter=60, n_per_block=n_per)

print("2. tutorial 10: + water R2 factor, M_start and B0 spread, shared by the seven tubes ...")
Kp, Fp, T2p = (replace(s, samples=pool) for s in (K, F, T2))
spec10 = FitSpec([Kp, Fp, T2p, R2F, MS, SIG])
x0 = np.tile([s.x0 for s in spec10.slots], (len(names), 1))
x0[list(pool), :3] = plain.x
ladder, Z = quesp(data)
water = fit_joint(ladder, spec10, Z, n_iter=12, n_per_block=n_per, x0_phys=x0, spread=spread, callback=progress)

print("3. the same + one B1 factor per rung, shared by the tubes ...")
x1 = np.column_stack([water.x, np.ones(len(names))])          # from step 2; the B1 factors start at 1
rung = fit_joint(ladder, FitSpec([Kp, Fp, T2p, R2F, MS, SIG, B1R]), Z, n_iter=12, n_per_block=n_per,
                 x0_phys=x1, spread=spread, block_jacobian="split", callback=progress)

b1 = rung.blocks["b1"]
pbs = [i for i in range(len(names)) if i not in pool]
print(f"\n{'rung':>8} {'B1 factor':>14} {'RMS pool tubes (10 -> +B1)':>28} {'RMS PBS (10 -> +B1)':>22}")
for j, b in enumerate(levels):
    flag = "  at bound" if b1["at_bounds"][j] else ""
    print(f"{b:6.2f} uT {b1['x'][j]:8.3f} +- {b1['sd'][j]:.3f} "
          f"{water.rms_per_block[list(pool), j].mean():>13.5f} -> {rung.rms_per_block[list(pool), j].mean():.5f} "
          f"{water.rms_per_block[pbs, j].mean():>11.5f} -> {rung.rms_per_block[pbs, j].mean():.5f}{flag}")
print(f"mean factor {b1['x'].mean():.3f}; water R2 x{water['r2w'][0]:.3f} -> x{rung['r2w'][0]:.3f}; "
      f"M_start {water['m_start'][0]:.3f} -> {rung['m_start'][0]:.3f}; "
      f"spread {water['sigma'][0]:.2f} -> {rung['sigma'][0]:.2f} Hz")

print()
print_pairs(water["f"], names, "tutorial 10")
print_pairs(rung["f"], names, "+ B1 per rung")
for i in pool:
    print(f"  {names[i]:<18} k {water['k'][i]:7.1f} -> {rung['k'][i]:7.1f}   f {water['f'][i]:.5f} -> {rung['f'][i]:.5f}")

# Eight more free values always lower the misfit somewhat. The per-rung B1 is worth keeping only if the
# drop sits where B1 matters (the strong rungs), PBS improves with the pool tubes, the factors are physical
# (no bound, a few percent from 1, smooth with power) and the checks the fit never saw -- the pairs, QUESP
# against QUEST -- hold.
