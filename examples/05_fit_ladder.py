"""05 -- Several saturation powers: the QUESP ladder.

Tutorial 04 showed that one saturation level measures k well only where the saturation
efficiency alpha happens to be in a useful range, and badly at both ends. A ladder fixes that by
giving the fit several values of alpha for the same pool -- no single rung has to be well
conditioned, because the SHAPE across the rungs carries the information.

Run:  python 05_fit_ladder.py
"""
import numpy as np

from bmc_jax.fitting import FitSpec, Problem, Slot, fit
from bmc_jax.io import example_config
from bmc_jax.jax_solver import convert_params, cw_saturation, get_initial_magnetization

print(__doc__)

params = example_config("amide")
jax_params = convert_params(params)
m0 = get_initial_magnetization(jax_params)
ppm = np.linspace(-6.0, 6.0, 61)
LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
NOISE = 0.002

# ---------------------------------------------------------------------------------------------
# Build the ladder
# ---------------------------------------------------------------------------------------------
# Problem.stack joins the rungs along the offset axis, so ONE residual spans the whole ladder.
# That is what makes k and f identifiable. The measured data must be concatenated in the SAME
# order -- that is the caller's job and the easiest thing here to get wrong, because getting it
# wrong does not raise, it just produces a fit.

one_rung = Problem(jax_params, cw_saturation(ppm, b1_uT=1.0, tsat_s=2.0, b0_T=params.system.b0),
                   m0, params.mz_loc)
ladder = Problem.stack([
    Problem(jax_params, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0), m0,
            params.mz_loc)
    for b in LADDER])
print(f"one rung: {one_rung.n_offsets} offsets    ladder: {ladder.n_offsets} offsets "
      f"({len(LADDER)} x {len(ppm)})")


def spec(x0_k):
    return FitSpec([Slot("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=x0_k),
                    Slot("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0003)])


def measure(problem, k_true, seed=0):
    truth = jax_params._replace(cest_k=np.float32([k_true]))
    z = np.asarray(Problem(truth, problem.events, get_initial_magnetization(truth),
                           params.mz_loc).simulate())
    return (z + np.random.default_rng(seed).normal(0, NOISE, z.shape))[None, :]


# ---------------------------------------------------------------------------------------------
# Same noise, same starting values, one rung against five
# ---------------------------------------------------------------------------------------------
print(f"\nwith {NOISE:.1%} noise, started 3x away from truth:")
print(f"{'k true':>8} | {'one rung':>10} {'err':>6} | {'5-rung ladder':>14} {'err':>6}")
print("-" * 56)
for k_true in (50.0, 150.0, 500.0, 1500.0, 4000.0):
    row = []
    for prob in (one_rung, ladder):
        blank = Problem(jax_params, prob.events, m0, params.mz_loc)
        k_fit = fit(blank, spec(3 * k_true), measure(prob, k_true), n_iter=80)["k"][0]
        row += [k_fit, abs(k_fit / k_true - 1)]
    print(f"{k_true:>8.0f} | {row[0]:>10.0f} {row[1]:>5.0%} | {row[2]:>14.0f} {row[3]:>5.0%}")

print("""
  -> every rung improves, most dramatically where one level was worst. But note k = 50: still
     the hardest case at ~30 %, because it is slow enough that even the 0.25 uT rung saturates
     it substantially. A ladder widens the range of k you can measure; it does not make the
     range unlimited, and the bottom rung sets the slow end.""")

# ---------------------------------------------------------------------------------------------
# What the result tells you
# ---------------------------------------------------------------------------------------------
# n_per_block lets FitResult report the residual per rung. That matters: a model error that
# grows with saturation power is invisible in the pooled RMS and obvious per rung, because the
# pooled number averages it away with the rungs that fit well.

k_true = np.array([80.0, 400.0, 2000.0])
Z = np.concatenate([measure(ladder, k, seed=i) for i, k in enumerate(k_true)])
blank = Problem(jax_params, ladder.events, m0, params.mz_loc)
res = fit(blank, spec(300.0), Z, n_iter=80, n_per_block=len(ppm))

print(f"\n{res}\n")
print(res.summary())
print(f"\ntruth        {np.array2string(k_true, precision=0)}")
print(f"recovered    {np.array2string(res['k'], precision=0)}")

print(f"\nRMS per rung (B1 = {', '.join(f'{b}' for b in LADDER)} uT):")
for i, per in enumerate(res.rms_per_block):
    print(f"  sample {i}  " + "  ".join(f"{v:.2e}" for v in per))

print("""
res.at_bounds is worth checking every time. A parameter sitting on its bound is not a
measurement -- the optimiser wanted to leave the allowed range and was stopped -- and the value
reported is the bound, not the data's answer. summary() marks those rows with a star.

Next: 06_quest.py -- the other ladder, in saturation TIME rather than power.""")
