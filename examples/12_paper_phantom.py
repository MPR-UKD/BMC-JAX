"""Tutorial 12: Reproducing the phantom results of the paper

This tutorial repeats the phantom fits of the paper that describes BMC-JAX, with the data that ship in
examples/data/, and prints every fitted value next to the published one (Table 3 of the paper). It
uses the same model, the same start values, the same bounds and the same optimiser settings.

The model (one fit per experiment, all seven tubes together)
  per tube with a CEST pool   exchange rate k, pool size f and pool T2 (fitted as its rate R2)
  shared by all seven tubes   the start magnetisation M_start, one factor on the water R2 of the PBS
                              tube, and the width of the B0 distribution within a tube (sigma, Hz)
  held fixed                  the water R1 of each tube (mean of the maps at the start and the end of
                              the session), the relative B1 of each tube, the pool T1 (1 s) and the
                              chemical shifts (GAG 1.0, creatine 1.9, carnosine 3.2 ppm)
The PBS tube has no CEST pool and enters the fit through the shared water parameters only.

The recipe (identical for QUESP and QUEST)
  stage 1   without the B0 spread, from round start values: k 300 1/s, f 8e-3, pool T2 10 ms,
            M_start 0.5, water R2 factor 1. At most 60 steps, starting damping 1, so that the first
            steps from these distant start values are small.
  stage 2   everything free, with the B0 spread (15 Gauss-Hermite nodes, sigma bounded to 0 to 8 Hz,
            start 5 Hz), from the end point of stage 1. At most 40 steps, starting damping 0.01.
  both      stop when the damping exceeds 1e7 (no step lowers the cost any more); a value on its bound
            whose step would leave the bounds is held for that step; the Jacobian of a rejected step is
            reused; the Jacobian is built three tubes at a time; the matrix exponential is the plain
            float32 version (bmc_jax.settings.use_expm("plain")).

What to expect: the fits run in float32, which reproduces k and f of a well-determined tube to about
one percent (tutorial 07). Tubes whose parameters are poorly determined (creatine at pH 6.31 under
QUEST, both carnosine tubes) sit in a flat region of the cost, where rounding moves the end point
further. The paper reports carnosine at pH 6.59 as not determined.

Requirements: the core package; the data ship in examples/data/. Runtime: about 3.5 h on one
RTX 3070 for both experiments, most of it in stage 2 (15 simulations per tube and step).

Run:  python 12_paper_phantom.py              (QUESP, then QUEST)
      python 12_paper_phantom.py quesp        (one experiment only)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phantom import (EXCLUDE_PPM, load_data, load_events, print_pairs, quesp_stem, quest_stem,
                      start_magnetisation, vial_params)

import bmc_jax
from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit_joint
from bmc_jax.simulation import B0Spread

print(__doc__)

# ---------------------------------------------------------------------------------------------
# Published values (Table 3 of the paper): k (1/s), f (1e-3), pool T2 (ms), RMSE/sigma per tube,
# and the shared values. Carnosine at pH 6.59 is "not determined" in the paper; the values given
# here for it are those of the table's footnote, with the pool T2 on its lower bound of 1 ms.
# ---------------------------------------------------------------------------------------------
PAPER = {
    "QUESP": dict(tubes={"PBS": (None, None, None, 5.0),
                         "GAG100_pH6.31": (1220, 5.8, 3.3, 12), "GAG50_pH6.20": (990, 2.9, 5.4, 4.5),
                         "creatine_pH7.51": (550, 3.1, 32, 2.7), "creatine_pH6.31": (49, 3.1, 32, 3.9),
                         "carnosine_pH6.59": (9, 0.51, 1.0, 7.1), "carnosine_pH7.44": (57, 0.21, 15, 5.6)},
                  m_start=0.53, t2w=2.05, sigma=3.7),
    "QUEST": dict(tubes={"PBS": (None, None, None, 2.2),
                         "GAG100_pH6.31": (1260, 6.0, 2.8, 8.0), "GAG50_pH6.20": (1060, 3.1, 4.6, 3.4),
                         "creatine_pH7.51": (560, 3.2, 23, 2.5), "creatine_pH6.31": (32, 4.3, 36, 2.5),
                         "carnosine_pH6.59": (965, 0.04, 1.0, 3.5), "carnosine_pH7.44": (82, 0.15, 10, 3.0)},
                  m_start=0.54, t2w=2.02, sigma=8.0),
}

bmc_jax.settings.use_expm("plain")

data = load_data()                                       # all seven tubes, PBS first
names = [str(n) for n in data["vial_names"]]
n = len(names)
pool = tuple(int(i) for i in np.flatnonzero(np.isfinite(data["vial_dw"])))
keep = np.abs(data["ppm"]) >= EXCLUDE_PPM
n_per = int(keep.sum())

# The tubes as one batch of parameters, with the pool T2 at its start value of 10 ms. M_start enters as a
# factor on the start state at full magnetisation, so a value of 0.5 means half the equilibrium magnetisation.
params = vial_params(data, pool_t2_s=0.010)
m0_full = start_magnetisation(params, m_start=1.0)


def experiment(kind):
    """The stacked ladder of one experiment and its data, offsets within 1 ppm of water excluded."""
    if kind == "QUESP":
        stems, z = [quesp_stem(b) for b in data["b1_levels_uT"]], data["Z_quesp"]
    else:
        stems, z = [quest_stem(t) for t in data["tsat_s"]], data["Z_quest"]
    exp = Experiment.stack([Experiment(params, load_events(s, keep), m0_full, 4) for s in stems])
    return exp, z[:, :, keep].reshape(n, -1)


# ---------------------------------------------------------------------------------------------
# The free parameters: bounds, scales and start values of the paper
# ---------------------------------------------------------------------------------------------
K = FreeParam("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=2000.0, x0=300.0, samples=pool)
F = FreeParam("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008, samples=pool)
R2P = FreeParam("r2p", "cest_r2", 0, scale=100.0, lo=10.0, hi=1000.0, x0=100.0, samples=pool)
R2W = FreeParam("r2w", "water_r2", None, scale=1.0, lo=0.5, hi=2.0, x0=1.0, scope="shared", mode="factor")
MS = FreeParam("m_start", "m_start", None, scale=1.0, lo=0.0, hi=1.0, x0=0.5, scope="shared", mode="factor")
SIG = FreeParam("sigma", "b0_spread", None, scale=10.0, lo=0.0, hi=8.0, x0=5.0, scope="shared")
RULES = dict(stop_lam=1e7, active_bounds=True, keep_jacobian=True, max_chunk=3)


def progress(t0):
    return lambda it, c: print(f"   step {it + 1:2d}  cost {c:.6g}  ({time.time() - t0:.0f} s)", flush=True)


def run(kind):
    exp, z = experiment(kind)
    print(f"\n{'=' * 100}\n{kind}: {n} tubes x {z.shape[1]} points ({z.shape[1] // n_per} rungs x {n_per} offsets)")

    t0 = time.time()
    print("stage 1: no B0 spread, from the round start values")
    s1 = fit_joint(exp, FitSpec([K, F, R2P, R2W, MS]), z, n_iter=60, n_per_block=n_per, lam0=1.0,
                   callback=progress(t0), **RULES)
    print(f"stage 1 done: {len(s1.cost)} steps, {time.time() - t0:.0f} s")

    t1 = time.time()
    print("stage 2: everything free, B0 spread with 15 nodes, from the end of stage 1")
    x0 = np.column_stack([s1.x, np.full(n, SIG.x0)])
    s2 = fit_joint(exp, FitSpec([K, F, R2P, R2W, MS, SIG]), z, n_iter=40, n_per_block=n_per, x0_phys=x0,
                   lam0=1e-2, spread=B0Spread(nodes=15), callback=progress(t1), **RULES)
    print(f"stage 2 done: {len(s2.cost)} steps, {time.time() - t1:.0f} s")
    return s1, s2


def report(kind, res):
    pub = PAPER[kind]
    noise = data["noise_quesp" if kind == "QUESP" else "noise_quest"]
    print(f"\n{kind}: this run against the paper (Table 3)")
    print(f"{'tube':<18} {'k (1/s)':>15} {'f (1e-3)':>15} {'pool T2 (ms)':>15} {'RMSE/sigma':>13}  on a bound")
    print(f"{'':<18} {'run':>7} {'paper':>7} {'run':>7} {'paper':>7} {'run':>7} {'paper':>7} {'run':>6} {'paper':>6}")
    print("-" * 104)
    for i, name in enumerate(names):
        p = pub["tubes"]["PBS" if i not in pool else name]
        rms = res.rms[i] / noise[i]
        if i not in pool:
            print(f"{name:<18} {'':>47} {rms:>6.1f} {p[3]:>6.1f}")
            continue
        k, f, t2 = res["k"][i], 1e3 * res["f"][i], 1e3 / res["r2p"][i]
        bound = ", ".join(s.name for s, b, fr in zip(res.spec.slots, res.at_bounds[i], res.free[i]) if b and fr) or "-"
        print(f"{name:<18} {k:>7.0f} {p[0]:>7.0f} {f:>7.2f} {p[1]:>7.2f} {t2:>7.1f} {p[2]:>7.1f} {rms:>6.1f} {p[3]:>6.1f}"
              f"  {bound}")
    r2w = res["r2w"][0]
    print(f"\nshared:  M_start {res['m_start'][0]:.3f} (paper {pub['m_start']:.2f})   "
          f"water T2 {1.0 / (float(data['water_r2']) * r2w):.2f} s (paper {pub['t2w']:.2f})   "
          f"sigma B0 {res['sigma'][0]:.2f} Hz (paper {pub['sigma']:.1f})")
    print()
    print_pairs(res["f"], names, f"{kind} pair checks")


kinds = [a.upper() for a in sys.argv[1:]] or ["QUESP", "QUEST"]
results = {}
for kind in kinds:
    s1, s2 = run(kind)
    results[kind] = s2
    report(kind, s2)
    np.savez(Path(__file__).with_name(f"12_paper_phantom_{kind.lower()}.npz"), names=names, x=s2.x,
             x_stage1=s1.x, rms=s2.rms, cost=s2.cost, cost_stage1=s1.cost, z_fit=s2.z_fit,
             at_bounds=s2.at_bounds)

print("""
How to read the comparison: the run and the paper used the same data, model and settings, but not the
same software build, so float32 rounding differs. Well-determined values agree to about one percent.
Values that the data do not determine well, such as the slow tubes and the B0 width in QUEST, may
differ more, and that difference is itself a measure of how poorly they are determined.""")
