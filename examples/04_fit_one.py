"""04 -- Fitting k and f from ONE saturation level, and when that is not enough.

The usual way this is taught is "one level cannot separate k from f, full stop". That is too
strong, and this tutorial shows why: on clean data a single level recovers both perfectly well.
What breaks it is the combination of NOISE and a badly conditioned operating point -- and the
condition has a name you can compute before you scan.

Run:  python 04_fit_one.py
"""
import numpy as np

from bmc_jax.fitting import FitSpec, Experiment, Slot, fit
from bmc_jax.io import example_config
from bmc_jax.jax_solver import convert_params, cw_saturation, get_initial_magnetization

print(__doc__)

params = example_config("amide")
jax_params = convert_params(params)
ppm = np.linspace(-6.0, 6.0, 61)
B1_UT = 1.0
NOISE = 0.002                       # 0.2 % of M0 -- an ordinary single-average CEST measurement

events = cw_saturation(ppm, b1_uT=B1_UT, tsat_s=2.0, b0_T=params.system.b0)
blank = Experiment(jax_params, events, get_initial_magnetization(jax_params), params.mz_loc)


def spec(x0_k):
    """Free k and f. `scale` puts each near 1 for the optimiser: k ~ 150 and f ~ 0.0009 differ
    by five orders of magnitude, and raw steps would be huge in one and invisible in the other."""
    return FitSpec([Slot("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=x0_k),
                    Slot("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0003)])


def measure(k_true, noise=0.0, seed=0):
    truth = jax_params._replace(cest_k=np.float32([k_true]))
    z = np.asarray(Experiment(truth, events, get_initial_magnetization(truth), params.mz_loc).simulate())
    if noise:
        z = z + np.random.default_rng(seed).normal(0, noise, z.shape)
    return z[None, :]


# ---------------------------------------------------------------------------------------------
# 1. On clean data, one level is fine
# ---------------------------------------------------------------------------------------------
print("CLEAN DATA, one saturation level, started deliberately 3x away from the truth:")
print(f"{'k true':>8} {'k fitted':>10} {'ratio':>7}")
for k_true in (50.0, 150.0, 500.0, 1500.0, 4000.0):
    k_fit = fit(blank, spec(3 * k_true), measure(k_true), n_iter=80)["k"][0]
    print(f"{k_true:>8.0f} {k_fit:>10.1f} {k_fit / k_true:>7.2f}x")
print("  -> the degeneracy is not absolute. Noiseless, the curvature is there to be used.")

# ---------------------------------------------------------------------------------------------
# 2. Add ordinary noise and it falls apart at the ends
# ---------------------------------------------------------------------------------------------
# alpha is the saturation efficiency -- what fraction of the labile pool the pulse actually
# saturates:
#
#     alpha = w1^2 / (w1^2 + k * (k + R2s))
#
# It is the lever the fit uses to tell k from f. Where alpha is already ~1 the pool is fully
# saturated and pushing k further changes nothing; where alpha is ~0 almost nothing is saturated
# and the effect is too small to shape. Either way the lever is short, and noise fills the gap.

r2s = 1.0 / params.cest_pools[0].T2
w1 = B1_UT * 267.5153

print(f"\nWITH {NOISE:.1%} NOISE, same fits:")
print(f"{'k true':>8} {'alpha':>7} {'k fitted':>10} {'error':>7}   why")
errors = {}
for k_true in (50.0, 150.0, 500.0, 1500.0, 4000.0):
    alpha = w1 ** 2 / (w1 ** 2 + k_true * (k_true + r2s))
    k_fit = fit(blank, spec(3 * k_true), measure(k_true, NOISE), n_iter=80)["k"][0]
    err = abs(k_fit / k_true - 1.0)
    errors[k_true] = err
    why = ("alpha near 1: the pool is already saturated, so raising k changes nothing"
           if alpha > 0.9 else
           "alpha near 0: almost nothing is saturated, so the effect is too small to shape"
           if alpha < 0.05 else "alpha in a useful range")
    print(f"{k_true:>8.0f} {alpha:>7.3f} {k_fit:>10.0f} {err:>6.0%}   {why}")

worst = max(errors, key=errors.get)
best = min(errors, key=errors.get)
print(f"""
  -> {errors[worst]:.0%} error at k = {worst:.0f} against {errors[best]:.0%} at k = {best:.0f},
     from the same data quality and the same fit. The difference is entirely conditioning.

The failure is quiet. Nothing rails, nothing warns, the fit converges, and the residual looks
fine. Compute alpha before you believe a k: the number is only as good as the lever that
measured it, and at one saturation power you get one lever.

Next: 05_fit_ladder.py -- several saturation powers give several levers.""")
