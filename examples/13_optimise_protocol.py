"""Tutorial 13: Optimising a protocol

Tutorial 12 computed the Cramer-Rao lower bound (CRLB): the smallest uncertainty with which a protocol
can measure k and f. This tutorial lets the computer choose a better protocol. It changes the
saturation amplitudes of a QUESP ladder until the bound of k and f is as small as possible.

The procedure
  1. Choose what may change, the design: here the amplitudes of four rungs, each between 0.1 and
     3 uT. The upper limit stands for the power limit of the scanner.
  2. Keep everything else fixed, in particular the scan time: four rungs of 2 s saturation with 61
     offsets each. Only then are the start protocol and the optimised protocol comparable.
  3. Choose the pools the protocol should work for. The bound depends on the true values, so the
     cost is averaged over a slow (k = 50 1/s) and a fast (k = 500 1/s) amide pool.
  4. Minimise the cost, the sum of the squared relative bounds of k and f, by gradient descent with
     bmc_jax.design.optimize_design. The gradient with respect to the amplitudes comes from automatic
     differentiation through the whole chain: amplitudes, events, simulation, Jacobian, bound.

Making the amplitude differentiable
  cw_saturation builds the events with NumPy, so its b1_uT argument cannot be differentiated. The
  amplitude enters the events linearly, however, so every rung is built once at 1 uT, and its rf_amp
  is multiplied by the design value. The function build(b1) does this with JAX operations only.

Precision: float64, as in tutorial 12.

Requirements: the core package only.

Run:  python 13_optimise_protocol.py
"""
from bmc_jax.settings import use_x64

use_x64(True)                      # before any array is created

import jax
import jax.numpy as jnp
import numpy as np

from bmc_jax.design import crlb, optimize_design
from bmc_jax.fitting import Experiment, FitSpec, FreeParam
from bmc_jax.io import example_config
from bmc_jax.sequences import cw_saturation
from bmc_jax.simulation import initial_state, to_sim_params

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
m0 = initial_state(jax_params)
ppm = np.linspace(-6.0, 6.0, 61)
F_TRUE = params.cest_pools[0].f
NOISE = 0.002

# ---------------------------------------------------------------------------------------------
# 1. The design and the function that turns it into an experiment
# ---------------------------------------------------------------------------------------------
N_RUNGS = 4
START = np.array([0.5, 1.0, 1.5, 2.0])     # uT, an evenly spaced ladder as a typical first choice
LO, HI = np.full(N_RUNGS, 0.1), np.full(N_RUNGS, 3.0)
rung_1uT = cw_saturation(ppm, b1_uT=1.0, tsat_s=2.0, b0_T=params.system.b0)


def build(b1):
    """The ladder at the amplitudes b1 (uT): every rung is the 1 uT rung with its amplitude scaled."""
    rungs = [rung_1uT._replace(rf_amp=rung_1uT.rf_amp * b1[j]) for j in range(N_RUNGS)]
    events = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), *rungs)
    return Experiment(jax_params, events, m0, params.mz_loc)


# The parameters that will be estimated, as for a fit, and the pools the protocol should work for.
spec = FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=150.0),
                FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=F_TRUE)])
TRUTHS = [[50.0, F_TRUE], [500.0, F_TRUE]]

# ---------------------------------------------------------------------------------------------
# 2. The bound of the start protocol
# ---------------------------------------------------------------------------------------------
print(f"Start protocol: {', '.join(f'{b:g}' for b in START)} uT, 2 s each, noise {NOISE:g}")
for x in TRUTHS:
    b = crlb(build(jnp.asarray(START)), spec, NOISE, x=x)
    print(f"  k = {x[0]:g} 1/s:  CRLB of k {100 * b.rel_sd[0]:.1f} %,  of f {100 * b.rel_sd[1]:.1f} %")

# ---------------------------------------------------------------------------------------------
# 3. Optimisation
# ---------------------------------------------------------------------------------------------
print("\nOptimising the four amplitudes (Adam, 150 steps):")


def progress(step, cost, design):
    if (step + 1) % 25 == 0:
        print(f"  step {step + 1:3d}  cost {cost:.5f}  amplitudes {np.array2string(np.sort(design), precision=2)} uT")


res = optimize_design(build, spec, START, LO, HI, NOISE, TRUTHS, n_steps=150, lr=0.05, callback=progress)

# ---------------------------------------------------------------------------------------------
# 4. The result
# ---------------------------------------------------------------------------------------------
print(f"\nAmplitudes (uT), sorted:  start {np.array2string(np.sort(res.start), precision=2)}"
      f"   optimised {np.array2string(np.sort(res.design), precision=2)}")
print(f"Cost: {res.cost[0]:.5f} -> {res.cost[-1]:.5f}")
print(f"\n{'true k':>8} {'':>4} {'CRLB k':>18} {'CRLB f':>18}")
print(f"{'(1/s)':>8} {'':>4} {'start':>8} {'optimised':>9} {'start':>8} {'optimised':>9}")
for x, s, e in zip(res.truths, res.rel_sd_start, res.rel_sd_end):
    print(f"{x[0]:>8g} {'':>4} {100 * s[0]:>7.1f}% {100 * e[0]:>8.1f}% {100 * s[1]:>7.1f}% {100 * e[1]:>8.1f}%")

print("""
How to read the result
  * The scan time is the same (four rungs of 2 s), so the lower bound is a real improvement: the same
    measurement time gives a smaller uncertainty for the pools in TRUTHS.
  * The optimised ladder uses only two different amplitudes: a low one, and a high one repeated three
    times. This is typical of optimal designs. Two unknowns need only two well-chosen conditions, and
    repeating the more informative one adds data where it helps most.
  * The slow pool gains much and the fast pool loses a little. The cost adds the squared relative
    bounds, so the slow pool, whose bound is about ten times larger, dominates it. If the fast pool
    matters as much, optimise for it separately or change the pools in TRUTHS.
  * The optimum depends on the pools you ask for. A protocol optimised for slow exchange alone puts
    its amplitudes elsewhere than one for fast exchange; averaging over both gives a compromise.
  * Gradient descent finds a local optimum. Starting from another protocol, or with other limits
    (LO, HI), is a quick check that the result does not depend on the start.
  * The bound describes noise only. It does not include errors of the model or of the calibration
    (B0, B1, T1). When a fit is limited by such errors, as the measured phantom of tutorials 08 to 11
    is, a lower bound does not make the result more accurate. Check an optimised protocol with a
    Monte Carlo test (tutorial 12) and, in the end, with a measurement.""")
