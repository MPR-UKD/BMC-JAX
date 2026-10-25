"""Tutorial 10: Joint fit with water parameters shared by all tubes

In tutorial 08, every tube was fitted on its own, and three properties of the water were held
fixed: the water R2 (from the PBS tube), the start magnetisation M_start (fitted once to the PBS
tube alone), and a single frequency per tube, without a spread of B0 within the tube. All tubes
contain the same buffer and were measured with the same sequence in the same magnet, so these three
properties are shared by all tubes. fit_joint() fits them from the spectra, as one value for all
seven tubes (PBS included), together with k, f and the pool T2 of every tube with a CEST pool:

  water R2    one factor applied to the PBS value of every tube    (scope="shared", mode="factor")
  M_start     the start magnetisation of the sequence               (field "m_start", scope="shared")
  B0 spread   the width (Hz) of the B0 distribution within a tube   (field "b0_spread", scope="shared")

The B0 spread is modelled with B0Spread(nodes=15): every tube is simulated at 15 frequencies around
its mean B0, and the spectra are averaged with Gauss-Hermite weights. The width is bounded at 8 Hz.

A shared parameter adds one column to the Jacobian and is constrained by all tubes at once. This is
what makes it measurable: a single tube cannot separate an error of the water R2 from its own k and
f. The PBS tube has no CEST pool and constrains the water parameters alone. Its k and f are not
fitted, which FreeParam(samples=...) expresses.

This tutorial
  1. runs the fit of tutorial 08 (water held fixed) to obtain start values;
  2. runs the joint fit, 12 Levenberg-Marquardt steps, with the shared water parameters free;
  3. compares the two fits tube by tube, the residual per amplitude, and the pair checks.

Requirements: the core package; the data ship in examples/data/. Runtime: about 2 h on one
RTX 3070, because the B0 spread costs 15 simulations per tube and step.

Run:  python 10_shared_water.py
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

data = load_data()                                       # all seven tubes, PBS first
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

# ---------------------------------------------------------------------------------------------
# 1. Tutorial 08's fit: k, f and pool T2 per tube, the water held
# ---------------------------------------------------------------------------------------------
print("1. k, f and pool T2 per tube, the water held (tutorial 08) ...")
ladder6, Z6 = quesp(load_data(tubes=pool))
plain = fit(ladder6, FitSpec([K, F, T2]), Z6, n_iter=60, n_per_block=n_per)

# ---------------------------------------------------------------------------------------------
# 2. The same fit, with the shared water parameters free
# ---------------------------------------------------------------------------------------------
# The shared parameters start at the values that tutorial 08 held fixed: a factor of 1 on the PBS
# R2, and M_start from the PBS tube. The pool parameters are free in the pool tubes only and start
# at the result of step 1.
#
# The spread starts at 5 Hz and is bounded at 8 Hz. On this phantom the cost hardly changes with
# it: about 3.5 Hz and 8 Hz fit equally well, and only k and f of the slowest tube change with it.
# The fitted width is therefore an effective width, not a measurement. With 15 nodes, the averaging
# is accurate to a small fraction of the noise up to 8 Hz; fewer nodes are not
# (see bmc_jax.simulation.spread).
Kp, Fp, T2p = (replace(s, samples=pool) for s in (K, F, T2))
R2F = FreeParam("r2w", "water_r2", None, scale=1.0, lo=0.5, hi=2.0, x0=1.0, scope="shared", mode="factor")
MS = FreeParam("m_start", "m_start", None, scale=1.0, lo=0.2, hi=1.0, x0=M_START, scope="shared")
SIG = FreeParam("sigma", "b0_spread", None, scale=10.0, lo=0.0, hi=8.0, x0=5.0, scope="shared")
spread = B0Spread(nodes=15)
spec = FitSpec([Kp, Fp, T2p, R2F, MS, SIG])
x0 = np.tile([s.x0 for s in spec.slots], (len(names), 1))
x0[list(pool), :3] = plain.x                                       # k, f, pool R2 from step 1
ladder, Z = quesp(data)
print("2. the same fit, with water R2 factor, M_start and B0 spread shared by the seven tubes ...")
joint = fit_joint(ladder, spec, Z, n_iter=12, n_per_block=n_per, x0_phys=x0, spread=spread, max_chunk=3,
                  callback=lambda it, c: print(f"   step {it + 1:2d}  cost {c:.5g}", flush=True))

# ---------------------------------------------------------------------------------------------
# 3. Comparison of the two fits
# ---------------------------------------------------------------------------------------------
print(f"\nshared: water R2 x{joint['r2w'][0]:.3f} +- {joint.sd[0, 3]:.3f}   "
      f"M_start {joint['m_start'][0]:.3f} +- {joint.sd[0, 4]:.3f}   "
      f"B0 spread {joint['sigma'][0]:.2f} +- {joint.sd[0, 5]:.2f} Hz (15 nodes)")

noise = data["noise_quesp"]
print(f"\n{'tube':<18} {'k held':>7} {'shared':>7} {'f held':>8} {'shared':>8} {'T2 held':>8} {'shared':>7} "
      f"{'rms/noise held':>15} {'shared':>7}")
print("-" * 96)
for i, name in enumerate(names):
    if i not in pool:
        print(f"{name:<18} {'(no pool: water only)':>58} {'-':>15} {joint.rms[i] / noise[i]:>7.1f}")
        continue
    j = pool.index(i)
    print(f"{name:<18} {plain['k'][j]:>7.0f} {joint['k'][i]:>7.0f} {plain['f'][j]:>8.5f} {joint['f'][i]:>8.5f} "
          f"{1e3 / plain['r2p'][j]:>6.1f}ms {1e3 / joint['r2p'][i]:>5.1f}ms "
          f"{plain.rms[j] / noise[i]:>15.1f} {joint.rms[i] / noise[i]:>7.1f}")

print("\nRMS per rung, mean of the six pool tubes (water held, then shared):")
for b, a, c in zip(levels, plain.rms_per_block.mean(0), joint.rms_per_block[list(pool)].mean(0)):
    print(f"  {b:4.2f} uT   {a:.5f} -> {c:.5f}")

print()
print_pairs(plain["f"], [names[i] for i in pool], "held")
print_pairs(joint["f"], names, "shared")
on_bound = [f"{names[i]} {s.name}" for i in range(len(names)) for l, s in enumerate(spec.slots)
            if joint.at_bounds[i, l] and joint.free[i, l]]
print(f"cost from {joint.cost[0]:.4g} to {joint.cost[-1]:.4g}; on a bound: {', '.join(on_bound) or 'nothing'}")

# A lower misfit alone does not show a better model: three additional free parameters always fit
# somewhat better. What matters is that the shared values are physically plausible (R2 close to the
# PBS value, M_start between 0 and 1, a spread of a few Hz), that the PBS tube, which contains only
# water, fits better, and that the checks the fit did not use (the pair ratios here, and QUESP
# against QUEST) do not get worse.

print("\nNext: 11_per_rung.py, one parameter per rung of the ladder, shared by all tubes.")
