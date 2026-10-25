"""Tutorial 04: Fitting the exchange rate and pool size from one saturation level

The exchange rate k and the pool size f both change the depth of a CEST dip: a deeper dip can come
from a larger pool or from faster exchange. Whether a fit can separate the two depends on the
saturation efficiency

    alpha = w1^2 / (w1^2 + k (k + R2s)),

the fraction of the pool that the saturation actually saturates. Here w1 = gamma B1 is the
saturation amplitude in rad/s and R2s = 1/T2 is the transverse relaxation rate of the pool.

This tutorial
  1. explains the Jacobian, the matrix of derivatives that every fit step uses, and how BMC-JAX
     computes it: by automatic differentiation in forward mode or reverse mode, compared here with
     finite differences;
  2. fits k and f from noise-free spectra at five exchange rates (50 to 4000 1/s), with the start
     value of k three times the true value. Without noise, one saturation level recovers both;
  3. repeats the fits with 0.2 % noise. Where alpha is close to 1 (slow exchange: the pool is
     already fully saturated) or close to 0 (fast exchange: the pool is hardly saturated), the
     fitted k becomes unreliable, although the fit converges and the residual looks normal.

Conclusion: compute alpha for your protocol before relying on a fitted k. Tutorial 05 shows how
several saturation amplitudes extend the range of exchange rates that can be measured, and
tutorial 12 computes the smallest uncertainty a protocol allows (the Cramer-Rao lower bound).

The Jacobian
  The fit compares the measured spectrum Z with the simulated one and changes the parameters to
  reduce the difference. To know in which direction to change them, it needs the Jacobian J: one
  row per offset and one column per free parameter, with J[i, j] = dZ_i / d(theta_j), the change
  of the signal at offset i per unit change of parameter j. Levenberg-Marquardt computes every step
  from J.

  BMC-JAX obtains J by automatic differentiation (AD): JAX differentiates the simulation code
  itself, step by step, which gives the derivative of the computed spectrum to rounding precision
  and without a step size. AD has two modes:
    forward mode (jax.jacfwd)  carries the derivative along with the simulation: one pass per
                               parameter, and no need to store the course of the simulation;
    reverse mode (jax.jacrev)  runs the simulation, stores its course, and then goes back through
                               it: one pass per output (offset), with memory that grows with the
                               length of the sequence.
  Both give the same Jacobian, up to rounding. A CEST fit has few parameters (two or three) and many
  offsets (hundreds), so forward mode is cheaper in time and memory. fit() chooses forward mode
  automatically when there are more offsets than parameters, and fit_joint() always uses it.

  Without AD, the Jacobian is estimated by finite differences: each parameter is changed by a small
  step h and the simulation is repeated. The result depends on h: a large step gives a truncation
  error, a small step amplifies the rounding error of float32.

Fit setup: FitSpec lists the free parameters. Each FreeParam gives the parameter field it changes
(cest_k, cest_f), the pool index, a scale (the typical size of the parameter, so that the optimiser
works with numbers close to 1), the lower and upper bound, and the start value.

Requirements: the core package only.

Run:  python 04_fit_one.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from bmc_jax.fitting import FitSpec, Experiment, FreeParam, fit
from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state
from bmc_jax.sequences import cw_saturation

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
ppm = np.linspace(-6.0, 6.0, 61)
B1_UT = 1.0
NOISE = 0.002                       # 0.2 % of M0, typical of a single-average CEST measurement

events = cw_saturation(ppm, b1_uT=B1_UT, tsat_s=2.0, b0_T=params.system.b0)
blank = Experiment(jax_params, events, initial_state(jax_params), params.mz_loc)


def spec(x0_k):
    """Free k and f. `scale` brings both close to 1 for the optimiser: k ~ 150 and f ~ 0.0009 differ
    by five orders of magnitude, and unscaled steps would be far too large in one parameter and
    negligible in the other."""
    return FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=x0_k),
                    FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0003)])


def measure(k_true, noise=0.0, seed=0):
    truth = jax_params._replace(cest_k=np.float32([k_true]))
    z = np.asarray(Experiment(truth, events, initial_state(truth), params.mz_loc).simulate())
    if noise:
        z = z + np.random.default_rng(seed).normal(0, noise, z.shape)
    return z[None, :]


# ---------------------------------------------------------------------------------------------
# 1. The Jacobian, and how it is computed
# ---------------------------------------------------------------------------------------------
# The model the fit inverts: (k, f) in physical units -> Z-spectrum. The start state is held fixed,
# as in the fit, so this is exactly the function whose Jacobian the fit uses.
m0_fit = initial_state(jax_params)


def z_of(theta):
    """Z-spectrum for theta = (k, f)."""
    p = jax_params._replace(cest_k=jnp.reshape(theta[0], (1,)), cest_f=jnp.reshape(theta[1], (1,)))
    return Experiment(p, events, m0_fit, params.mz_loc).simulate()


theta = jnp.array([150.0, params.cest_pools[0].f], jnp.float32)
J_fwd = np.asarray(jax.jacfwd(z_of)(theta))           # forward mode: one pass per parameter
J_rev = np.asarray(jax.jacrev(z_of)(theta))           # reverse mode: one pass per offset
scale = np.abs(J_fwd).max(axis=0)
print(f"Jacobian at k = 150 1/s and f = {float(theta[1]):.4f}: {J_fwd.shape[0]} offsets x {J_fwd.shape[1]} parameters")
print(f"  forward and reverse mode differ by {np.max(np.abs(J_fwd - J_rev) / scale):.1e} (relative to each column's maximum)")

# Sensitivity per relative change: theta_j * dZ/dtheta_j is the change of Z for a 100 % change of the
# parameter, which makes the two columns comparable.
print(f"\n  {'offset':>8} {'k dZ/dk':>10} {'f dZ/df':>10}")
for target in (-3.5, 1.0, 2.6, 3.0, 3.4, 3.8, 4.4):
    i = int(np.argmin(np.abs(ppm - target)))
    print(f"  {ppm[i]:+6.1f} ppm {float(theta[0]) * J_fwd[i, 0]:>10.4f} {float(theta[1]) * J_fwd[i, 1]:>10.4f}")
cos = J_fwd[:, 0] @ J_fwd[:, 1] / (np.linalg.norm(J_fwd[:, 0]) * np.linalg.norm(J_fwd[:, 1]))
print(f"  Both columns are largest around the pool's shift of +3.5 ppm. Their shapes agree to {abs(cos):.4f}")
print("  (1 would be identical): a change of k can almost be compensated by a change of f. Tutorial 12")
print("  turns this into the smallest uncertainty the measurement allows.")

print("\n  Finite differences for dZ/dk with a central step of relative size h, against forward mode:")
for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    dk = float(theta[0]) * h
    fd = (np.asarray(z_of(theta.at[0].add(dk))) - np.asarray(z_of(theta.at[0].add(-dk)))) / (2 * dk)
    print(f"    h = {h:.0e}   error {np.max(np.abs(fd - J_fwd[:, 0])) / scale[0]:.1e}")
print("""
  Why the two AD modes differ by about 1e-4, and why finite differences fail here: the 2 s saturation
  is one long step whose matrix exponential is computed by repeated squaring, which amplifies float32
  rounding to about 1e-4 of the simulated signal. Forward and reverse mode both differentiate this
  computation exactly and differ only by that rounding. A finite difference divides the rounding by the
  step: a small step gives an error larger than the derivative itself, and even the largest step tested
  is several percent off. In float64 (tutorial 12) the two AD modes agree to about 1e-14 and finite
  differences come close at an intermediate step. In float32, the precision used for fast fitting on a
  GPU, automatic differentiation is the only reliable way to obtain this Jacobian.
""")

# ---------------------------------------------------------------------------------------------
# 2. Noise-free data
# ---------------------------------------------------------------------------------------------
print("Noise-free data, one saturation level, start value of k three times the true value:")
print(f"{'k true':>8} {'k fitted':>10} {'ratio':>7}")
for k_true in (50.0, 150.0, 500.0, 1500.0, 4000.0):
    k_fit = fit(blank, spec(3 * k_true), measure(k_true), n_iter=80)["k"][0]
    print(f"{k_true:>8.0f} {k_fit:>10.1f} {k_fit / k_true:>7.2f}x")
print("  Without noise, k and f are recovered from a single saturation level.")

# ---------------------------------------------------------------------------------------------
# 3. Data with 0.2 % noise
# ---------------------------------------------------------------------------------------------
# alpha is the saturation efficiency, the fraction of the labile pool that the pulse saturates:
#
#     alpha = w1^2 / (w1^2 + k * (k + R2s))
#
# It is what allows the fit to separate k from f. Where alpha is close to 1, the pool is already
# fully saturated and a faster k changes the spectrum very little. Where alpha is close to 0,
# almost nothing is saturated and the effect is small. In both cases the change of the spectrum
# with k is small compared with the noise.

r2s = 1.0 / params.cest_pools[0].T2
w1 = B1_UT * 267.5153

print(f"\nWith {NOISE:.1%} noise, the same fits:")
print(f"{'k true':>8} {'alpha':>7} {'k fitted':>10} {'error':>7}   why")
errors = {}
for k_true in (50.0, 150.0, 500.0, 1500.0, 4000.0):
    alpha = w1 ** 2 / (w1 ** 2 + k_true * (k_true + r2s))
    k_fit = fit(blank, spec(3 * k_true), measure(k_true, NOISE), n_iter=80)["k"][0]
    err = abs(k_fit / k_true - 1.0)
    errors[k_true] = err
    why = ("alpha near 1: the pool is already saturated, so a faster k changes little"
           if alpha > 0.9 else
           "alpha near 0: the pool is hardly saturated, so the effect is small"
           if alpha < 0.05 else "alpha in a useful range")
    print(f"{k_true:>8.0f} {alpha:>7.3f} {k_fit:>10.0f} {err:>6.0%}   {why}")

worst = max(errors, key=errors.get)
best = min(errors, key=errors.get)
print(f"""
  The error is {errors[worst]:.0%} at k = {worst:.0f} and {errors[best]:.0%} at k = {best:.0f}, with the same
  noise and the same fit. The difference comes only from the saturation efficiency alpha.

The fit gives no warning: it converges, no parameter reaches a bound, and the residual looks normal.
Compute alpha before relying on a fitted k. One saturation amplitude gives one value of alpha.

Next: 05_fit_ladder.py, several saturation amplitudes and therefore several values of alpha.""")
