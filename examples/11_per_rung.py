"""Tutorial 11: One parameter per saturation amplitude, shared by all tubes

Some errors belong to a rung of the ladder, not to a tube. The scanner plays each QUESP amplitude
at its own RF power. If the RF amplifier is not perfectly linear, the B1 delivered at one amplitude
can differ from the nominal value by a few percent. This deviation is the same in every tube, and
a B1 map measured at a single power does not show it. A slow B0 drift during the measurement has
a similar effect: one frequency offset per rung, the same in every tube.

FreeParam(scope="block") fits one value per experiment of a stacked ladder, shared by all samples:

  B1 per rung   field "rel_b1", mode="factor": the measured relative B1 of every tube is multiplied
                by the factor of the rung
  B0 per rung   field "b0_inhom", mode="add" would describe a drift. It is not fitted here, because
                the data are already B0-corrected.

This tutorial
  1. runs the fit of tutorial 08 to obtain start values;
  2. runs the joint fit of tutorial 10 (k, f and pool T2 per pool tube; water R2 factor, M_start
     and B0 spread shared by all seven tubes);
  3. runs the same joint fit with one additional B1 factor per rung, and compares the two fits rung
     by rung. block_jacobian="split" builds the eight B1 columns of the Jacobian from one
     forward-mode pass instead of eight; the derivative is the same.

How to read the result
  * Factors close to 1 without a trend: the amplifier is linear, and a B1 factor per rung explains
    nothing.
  * A smooth trend with the amplitude: a possible amplifier effect. A B1 map measured at a second
    power would test it.
  * The mean of the factors trades off against the water R2 factor, because both deepen the direct
    water saturation. Read the two together.
  * The PBS tube contains only water. A real B1 error changes its saturation as well, so a real
    per-rung B1 effect must also improve the PBS fit at the affected rungs. If only the tubes with a
    CEST pool improve, the factors compensate an error of the pool model and do not measure B1.

Requirements: the core package; the data ship in examples/data/. Runtime: about 3.5 h on one
RTX 3070 (two joint fits with 15 nodes).

Run:  python 11_per_rung.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phantom import (EXCLUDE_PPM, M_START, POOL_T2_START, load_data, load_events, print_pairs, quesp_stem,
                      start_magnetisation, vial_params)

import bmc_jax
from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit, fit_joint
from bmc_jax.simulation import B0Spread

# The plain matrix exponential (docs/guide.md, "Matrix exponential"). With the library version, the batched
# Jacobian of these long scanner sequences came back NaN on an RTX 3070, every step of the fit was rejected,
# and the fit ended at its start values without an error. The paper's fits used the plain version too.
bmc_jax.settings.use_expm("plain")

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

print("1. k, f and pool T2 per tube (tutorial 08), for the start values ...")
ladder6, Z6 = quesp(load_data(tubes=pool))
plain = fit(ladder6, FitSpec([K, F, T2]), Z6, n_iter=60, n_per_block=n_per)

print("2. tutorial 10: water R2 factor, M_start and B0 spread shared by the seven tubes ...")
Kp, Fp, T2p = (replace(s, samples=pool) for s in (K, F, T2))
spec10 = FitSpec([Kp, Fp, T2p, R2F, MS, SIG])
x0 = np.tile([s.x0 for s in spec10.slots], (len(names), 1))
x0[list(pool), :3] = plain.x
ladder, Z = quesp(data)
water = fit_joint(ladder, spec10, Z, n_iter=12, n_per_block=n_per, x0_phys=x0, spread=spread, max_chunk=3, callback=progress)

print("3. the same with one B1 factor per rung, shared by the tubes ...")
x1 = np.column_stack([water.x, np.ones(len(names))])          # from step 2; the B1 factors start at 1
rung = fit_joint(ladder, FitSpec([Kp, Fp, T2p, R2F, MS, SIG, B1R]), Z, n_iter=12, n_per_block=n_per,
                 x0_phys=x1, spread=spread, max_chunk=3, block_jacobian="split", callback=progress)

b1 = rung.blocks["b1"]
pbs = [i for i in range(len(names)) if i not in pool]
print(f"\n{'rung':>8} {'B1 factor':>14} {'RMS pool tubes (10, +B1)':>28} {'RMS PBS (10, +B1)':>22}")
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

# Eight additional free parameters always lower the misfit somewhat. A B1 factor per rung is worth
# keeping only if the misfit drops where B1 matters (the high amplitudes), the PBS tube improves together
# with the pool tubes, the factors are physically plausible (not on a bound, a few percent from 1, smooth
# with the amplitude), and the checks the fit did not use (the pair ratios, QUESP against QUEST) still hold.
