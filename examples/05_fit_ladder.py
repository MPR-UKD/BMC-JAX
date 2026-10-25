"""Tutorial 05: The QUESP ladder (several saturation amplitudes)

QUESP (quantification of exchange as a function of saturation power) measures the same sample at
several saturation amplitudes, the "rungs" of a ladder. Tutorial 04 showed that a single amplitude
measures k well only where the saturation efficiency alpha lies in a useful range. Each amplitude
gives a different alpha, and the change of the CEST effect from rung to rung separates the exchange
rate from the pool size. No single rung has to be well conditioned on its own.

This tutorial
  1. stacks five experiments (B1 = 0.25 to 4 uT) with Experiment.stack(), so that one residual
     spans the whole ladder;
  2. compares the k fitted from one amplitude with the k fitted from the five-rung ladder, with the
     same 0.2 % noise, at five exchange rates;
  3. fits three samples at once and shows how to read the result: the summary table, the residual
     per rung (rms_per_block) and the parameters on a bound (at_bounds).

Note: the measured spectra must be concatenated in the same order as the stacked experiments. A
wrong order raises no error; it only gives a wrong fit.

Requirements: the core package only.

Run:  python 05_fit_ladder.py
"""
import numpy as np

from bmc_jax.fitting import FitSpec, Experiment, FreeParam, fit
from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state
from bmc_jax.sequences import cw_saturation

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
m0 = initial_state(jax_params)
ppm = np.linspace(-6.0, 6.0, 61)
LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
NOISE = 0.002

# ---------------------------------------------------------------------------------------------
# Build the ladder
# ---------------------------------------------------------------------------------------------
# Experiment.stack joins the rungs along the offset axis, so that one residual spans the whole
# ladder. This joint residual is what makes k and f separable. The measured data must be
# concatenated in the same order. This is the responsibility of the caller: a wrong order raises
# no error, it only gives a wrong fit.

one_rung = Experiment(jax_params, cw_saturation(ppm, b1_uT=1.0, tsat_s=2.0, b0_T=params.system.b0),
                   m0, params.mz_loc)
ladder = Experiment.stack([
    Experiment(jax_params, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0), m0,
            params.mz_loc)
    for b in LADDER])
print(f"one rung: {one_rung.n_offsets} offsets    ladder: {ladder.n_offsets} offsets "
      f"({len(LADDER)} x {len(ppm)})")


def spec(x0_k):
    return FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=x0_k),
                    FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0003)])


def measure(problem, k_true, seed=0):
    truth = jax_params._replace(cest_k=np.float32([k_true]))
    z = np.asarray(Experiment(truth, problem.events, initial_state(truth),
                           params.mz_loc).simulate())
    return (z + np.random.default_rng(seed).normal(0, NOISE, z.shape))[None, :]


# ---------------------------------------------------------------------------------------------
# One amplitude against five, with the same noise and the same start values
# ---------------------------------------------------------------------------------------------
print(f"\nwith {NOISE:.1%} noise, start value of k three times the true value:")
print(f"{'k true':>8} | {'one rung':>10} {'err':>6} | {'5-rung ladder':>14} {'err':>6}")
print("-" * 56)
for k_true in (50.0, 150.0, 500.0, 1500.0, 4000.0):
    row = []
    for prob in (one_rung, ladder):
        blank = Experiment(jax_params, prob.events, m0, params.mz_loc)
        k_fit = fit(blank, spec(3 * k_true), measure(prob, k_true), n_iter=80)["k"][0]
        row += [k_fit, abs(k_fit / k_true - 1)]
    print(f"{k_true:>8.0f} | {row[0]:>10.0f} {row[1]:>5.0%} | {row[2]:>14.0f} {row[3]:>5.0%}")

print("""
  The ladder strongly reduces the error of k where a single amplitude was poor. Where a single
  amplitude is already well conditioned (k = 500), both give a similar error. The slowest pool
  (k = 50) remains the most difficult case, because even the lowest amplitude (0.25 uT) saturates
  it substantially. A ladder extends the range of exchange rates that can be measured, and its
  lowest amplitude sets the lower end of that range.""")

# ---------------------------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------------------------
# n_per_block lets FitResult report the residual per rung. A model error that grows with the
# saturation amplitude is hidden in the overall RMS, which averages it with the rungs that fit
# well, but it is clearly visible in the RMS per rung.

k_true = np.array([80.0, 400.0, 2000.0])
Z = np.concatenate([measure(ladder, k, seed=i) for i, k in enumerate(k_true)])
blank = Experiment(jax_params, ladder.events, m0, params.mz_loc)
res = fit(blank, spec(300.0), Z, n_iter=80, n_per_block=len(ppm))

print(f"\n{res}\n")
print(res.summary())
print(f"\ntruth        {np.array2string(k_true, precision=0)}")
print(f"recovered    {np.array2string(res['k'], precision=0)}")

print(f"\nRMS per rung (B1 = {', '.join(f'{b}' for b in LADDER)} uT):")
for i, per in enumerate(res.rms_per_block):
    print(f"  sample {i}  " + "  ".join(f"{v:.2e}" for v in per))

print("""
Check res.at_bounds after every fit. A parameter on its bound is not a measurement: the optimiser
tried to leave the allowed range and was stopped, so the reported value is the bound and not the
value the data indicate. summary() marks these values with a star.

Next: 06_quest.py, the QUEST ladder, which varies the saturation time instead of the amplitude.""")
