"""Tutorial 12: The Jacobian and the Cramer-Rao lower bound

How precisely can a protocol measure an exchange rate? The Cramer-Rao lower bound (CRLB) answers
this before any measurement is made. It is the smallest standard deviation that any unbiased
estimate of a parameter can have, for a given protocol, a given noise level and given true values.
It is computed from the Jacobian of the model alone.

Background
  The Jacobian J of the model has one row per measured point (offset) and one column per parameter:
  J[i, j] = dZ_i / d(theta_j), the change of the signal at point i per unit change of parameter j
  (tutorial 04). For independent Gaussian noise of standard deviation sigma on every point, the
  Fisher information matrix is

      F = J^T J / sigma^2.

  It measures how strongly the data depend on the parameters. The CRLB is its inverse:

      Cov(theta) >= F^-1,   so   SD(theta_j) >= sqrt( (F^-1)[j, j] ).

  The off-diagonal elements of F^-1 give the correlation between the estimates. Two parameters
  whose Jacobian columns have almost the same shape, such as k and f of one CEST pool, are strongly
  correlated, and both have a large bound: the data can hardly tell them apart.

What the CRLB is used for
  * predicting the precision of a protocol before scanning;
  * comparing protocols, for example one saturation amplitude against a ladder of amplitudes;
  * checking that a fit reaches the precision the data allow.

This tutorial
  1. computes the Jacobian of a five-rung QUESP ladder for one amide pool, and compares forward
     mode, reverse mode and finite differences in float64 (tutorial 04 did this in float32);
  2. computes the Fisher information matrix, the CRLB of k and f, and their correlation;
  3. checks the bound with a Monte Carlo experiment: 200 copies of the same spectrum with different
     noise are fitted, and the spread of the fitted values is compared with the bound;
  4. uses the bound to compare protocols: each saturation amplitude alone against the full ladder,
     for a slow (k = 50 1/s) and a fast (k = 500 1/s) pool;
  5. lists what the bound does not include.

Precision: this tutorial runs in float64. The Fisher matrix of k and f is close to singular, and
float32 rounding in the Jacobian can change its inverse by a large factor (bmc_jax.settings).

Requirements: the core package only.

Run:  python 12_crlb.py
"""
from bmc_jax.settings import use_x64

use_x64(True)                      # before any array is created

import jax
import jax.numpy as jnp
import numpy as np

from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit
from bmc_jax.io import example_config
from bmc_jax.sequences import cw_saturation
from bmc_jax.simulation import initial_state, to_sim_params

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
m0 = initial_state(jax_params)
ppm = np.linspace(-6.0, 6.0, 61)
F_TRUE = params.cest_pools[0].f
R2S = 1.0 / params.cest_pools[0].T2
NOISE = 0.002                      # 0.2 % of M0, as in tutorials 04 to 06
LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)


def events_for(b1_list):
    """The stacked sequence of a list of saturation amplitudes (uT), 2 s continuous-wave saturation."""
    return Experiment.stack([Experiment(jax_params, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0),
                                        m0, params.mz_loc) for b in b1_list]).events


def model(events):
    """The model as a function of theta = (k, f) in physical units, with the start state held as in the fit."""
    def z_of(theta):
        p = jax_params._replace(cest_k=jnp.reshape(theta[0], (1,)), cest_f=jnp.reshape(theta[1], (1,)))
        return Experiment(p, events, m0, params.mz_loc).simulate()
    return z_of


def crlb(events, k, f=F_TRUE, sigma=NOISE):
    """Jacobian, Fisher matrix, CRLB (SD of k and f) and the k-f correlation at the true values (k, f)."""
    J = jax.jacfwd(model(events))(jnp.array([k, f]))
    fisher = J.T @ J / sigma ** 2
    cov = jnp.linalg.inv(fisher)
    sd = jnp.sqrt(jnp.diag(cov))
    return np.asarray(J), np.asarray(fisher), np.asarray(sd), float(cov[0, 1] / (sd[0] * sd[1]))


# ---------------------------------------------------------------------------------------------
# 1. The Jacobian of a QUESP ladder
# ---------------------------------------------------------------------------------------------
K_TRUE = 150.0
ladder = events_for(LADDER)
z_of = model(ladder)
theta = jnp.array([K_TRUE, F_TRUE])
J = np.asarray(jax.jacfwd(z_of)(theta))
J_rev = np.asarray(jax.jacrev(z_of)(theta))
scale = np.abs(J).max(axis=0)
print(f"1. Jacobian of the ladder B1 = {', '.join(f'{b:g}' for b in LADDER)} uT at k = {K_TRUE:g} 1/s, f = {F_TRUE:g}")
print(f"   shape {J.shape}: {len(LADDER)} rungs x {len(ppm)} offsets, and 2 parameters")
print(f"   forward and reverse mode differ by {np.max(np.abs(J - J_rev) / scale):.1e} (float64)")
print("   finite differences for dZ/dk, central step of relative size h:")
for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    dk = K_TRUE * h
    fd = (np.asarray(z_of(theta.at[0].add(dk))) - np.asarray(z_of(theta.at[0].add(-dk)))) / (2 * dk)
    print(f"     h = {h:.0e}   error {np.max(np.abs(fd - J[:, 0])) / scale[0]:.1e}")
print("   In float64, finite differences come close to the exact derivative at an intermediate step, where the")
print("   truncation error of large steps and the rounding error of small steps are balanced. Automatic")
print("   differentiation needs no step and is exact to rounding.\n")

# ---------------------------------------------------------------------------------------------
# 2. Fisher information and the Cramer-Rao lower bound
# ---------------------------------------------------------------------------------------------
_, fisher, sd, corr = crlb(ladder, K_TRUE)
print(f"2. Fisher information F = J^T J / sigma^2 with sigma = {NOISE:g}:")
print(f"     [[{fisher[0, 0]:10.4g}  {fisher[0, 1]:10.4g}]")
print(f"      [{fisher[1, 0]:10.4g}  {fisher[1, 1]:10.4g}]]")
print(f"   CRLB, the smallest possible standard deviation of an unbiased estimate:")
print(f"     k: {K_TRUE:g} +- {sd[0]:.2f} 1/s   ({100 * sd[0] / K_TRUE:.1f} %)")
print(f"     f: {F_TRUE:.5f} +- {sd[1]:.5f}   ({100 * sd[1] / F_TRUE:.1f} %)")
print(f"   correlation of the k and f estimates: {corr:+.3f}")
print("   A correlation close to -1 means that a higher k and a lower f give almost the same spectrum. The")
print("   bound includes this: each parameter alone would be known far better if the other were fixed.\n")

# ---------------------------------------------------------------------------------------------
# 3. Monte Carlo check: does a real fit reach the bound?
# ---------------------------------------------------------------------------------------------
# 200 copies of the same noise-free spectrum, each with new noise, are fitted from the true values.
# The spread of the 200 fitted values is the precision the fit actually achieves. For an efficient,
# unbiased estimator it equals the CRLB, within the Monte Carlo uncertainty of about 1/sqrt(2N) = 5 %.
N = 200
z_true = np.asarray(z_of(theta))
Z = z_true + np.random.default_rng(0).normal(0.0, NOISE, (N, z_true.size))
spec = FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=K_TRUE),
                FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=F_TRUE)])
res = fit(Experiment(jax_params, ladder, m0, params.mz_loc), spec, Z, n_iter=40, n_per_block=len(ppm))
k_fit, f_fit = res["k"], res["f"]
mc_corr = np.corrcoef(k_fit, f_fit)[0, 1]
print(f"3. Monte Carlo: {N} noisy spectra fitted")
print(f"   {'':6} {'mean':>10} {'SD of fits':>12} {'CRLB':>10} {'SD / CRLB':>10}")
print(f"   {'k':6} {k_fit.mean():>10.2f} {k_fit.std(ddof=1):>12.2f} {sd[0]:>10.2f} {k_fit.std(ddof=1) / sd[0]:>10.2f}")
print(f"   {'f':6} {f_fit.mean():>10.6f} {f_fit.std(ddof=1):>12.6f} {sd[1]:>10.6f} {f_fit.std(ddof=1) / sd[1]:>10.2f}")
print(f"   correlation of the fitted k and f: {mc_corr:+.3f} (CRLB: {corr:+.3f})")
print("   The fits reach the bound: their spread matches the CRLB, and the mean matches the true value. The")
print("   bound can therefore predict the precision of this protocol without fitting anything.\n")

# ---------------------------------------------------------------------------------------------
# 4. Comparing protocols with the bound
# ---------------------------------------------------------------------------------------------
# The relative CRLB of k for each amplitude alone, and for the five amplitudes together. alpha is
# the saturation efficiency of tutorial 04.
gamma = float(jax_params.gamma)
print("4. Relative CRLB of k and f, one amplitude alone against the whole ladder (same noise per point)")
for k in (50.0, 500.0):
    print(f"   k = {k:g} 1/s")
    print(f"     {'protocol':<16} {'alpha':>7} {'CRLB k':>9} {'CRLB f':>9} {'corr':>7}")
    for b in LADDER:
        w1 = gamma * b
        alpha = w1 ** 2 / (w1 ** 2 + k * (k + R2S))
        _, _, s, c = crlb(events_for((b,)), k)
        print(f"     {f'{b:g} uT':<16} {alpha:>7.3f} {100 * s[0] / k:>8.1f}% {100 * s[1] / F_TRUE:>8.1f}% {c:>+7.3f}")
    _, _, s, c = crlb(ladder, k)
    print(f"     {'ladder (all 5)':<16} {'':>7} {100 * s[0] / k:>8.1f}% {100 * s[1] / F_TRUE:>8.1f}% {c:>+7.3f}")
print("""   Where alpha is close to 1 (the pool is already saturated), a single amplitude gives a large bound and
   a correlation near -1: k and f cannot be separated. Where alpha is small, the bound grows as well,
   because the CEST effect itself becomes small. The best single amplitude lies in between. The ladder
   combines amplitudes with different alpha and gives the smallest bound, and it helps most for the
   slow pool, whose single amplitudes are all close to alpha = 1. The ladder also has five times as many
   points; to compare protocols of equal duration, scale the noise accordingly.
""")

# ---------------------------------------------------------------------------------------------
# 5. What the bound does not include
# ---------------------------------------------------------------------------------------------
print("""5. Limits of the CRLB
   * It is local: it is computed at assumed true values, and changes with them (section 4).
   * It assumes that the model is correct and the noise is known, Gaussian and independent.
   * It does not include systematic errors: a wrong B0, B1, T1, pool shift or an incomplete model.
   * It applies to unbiased estimates. Near a bound of the fit, or for poorly determined parameters,
     the fitted values become biased, and their spread can even fall below the CRLB.
   On measured data the misfit is usually larger than the noise (tutorials 08 to 11), so the real
   uncertainty is larger than the CRLB. The difference between two independent experiments, such as
   QUESP and QUEST, is then a more realistic estimate. fit_joint() reports error bars (res.sd) computed
   with the same formula, with sigma estimated from the misfit instead of the known noise.""")
