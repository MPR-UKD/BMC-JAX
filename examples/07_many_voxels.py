"""07 -- Fitting an image: many voxels at once, with per-voxel calibration.

Everything so far fitted one or three spectra. A slice is thousands, each with its own measured
T1, T2, B0 offset and B1 scaling. Those are FIXED per voxel -- they were measured, not fitted --
while k and f are free. This is what the batching exists for.

Run:  python 07_many_voxels.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from bmc_jax.fitting import FitSpec, Experiment, Slot, fit
from bmc_jax.io import example_config
from bmc_jax.jax_solver import (convert_params, cw_saturation, get_initial_magnetization,
                                simulate_spectrum, tile_params)

print(__doc__)

params = example_config("amide")
jax_params = convert_params(params)
ppm = np.linspace(-6.0, 6.0, 31)
N = 16 * 16                                   # a small "slice"
NOISE = 0.003

# ---------------------------------------------------------------------------------------------
# A phantom slice with spatially varying truth AND varying calibration
# ---------------------------------------------------------------------------------------------
# tile_params gives every voxel its own copy of the parameters, which you then overwrite per
# voxel. Anything you do not overwrite stays shared.

rng = np.random.default_rng(0)
k_true = np.linspace(80.0, 900.0, N).astype(np.float32)
b0_true = rng.normal(0.0, 0.05, N).astype(np.float32)        # ppm, a realistic shim residual
b1_true = rng.normal(1.0, 0.06, N).astype(np.float32)        # 6 % B1 variation across the slice
t1_true = rng.normal(1.3, 0.1, N).astype(np.float32)

base = tile_params(jax_params, N)
base = base._replace(cest_k=jnp.asarray(k_true[:, None]),
                     b0_inhom=jnp.asarray(b0_true),
                     rel_b1=jnp.asarray(b1_true),
                     water_r1=jnp.asarray(1.0 / t1_true))

ladder_b1 = (0.5, 1.0, 2.0)
events = Experiment.stack([
    Experiment(jax_params, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0),
            get_initial_magnetization(jax_params), params.mz_loc)
    for b in ladder_b1]).events

m0 = jnp.stack([get_initial_magnetization(jax.tree.map(lambda a, i=i: a[i], base))
                for i in range(N)])
Z = np.stack([np.asarray(simulate_spectrum(jax.tree.map(lambda a, i=i: a[i], base), events,
                                           m0[i], return_full_trajectory=False))[:, params.mz_loc]
              for i in range(N)])
Z = Z + rng.normal(0, NOISE, Z.shape)
print(f"slice: {N} voxels x {Z.shape[1]} offsets ({len(ladder_b1)} B1 rungs x {len(ppm)})")

# ---------------------------------------------------------------------------------------------
# Fit, giving the fitter the calibration it should NOT be guessing
# ---------------------------------------------------------------------------------------------
# The key point: b0_inhom, rel_b1 and water_r1 are per-voxel and stay FIXED at their measured
# values. Leaving them at a single slice-average instead is a common and expensive mistake --
# a 6 % B1 error is roughly doubled in the fitted k, because saturation efficiency goes as the
# square of the amplitude.

fit_base = tile_params(jax_params, N)._replace(
    b0_inhom=jnp.asarray(b0_true), rel_b1=jnp.asarray(b1_true),
    water_r1=jnp.asarray(1.0 / t1_true))
problem = Experiment(fit_base, events, m0, params.mz_loc)

spec = FitSpec([Slot("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=300.0),
                Slot("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0005)])

t0 = time.time()
res = fit(problem, spec, Z, n_iter=40, n_per_block=len(ppm))
print(f"\nfitted {N} voxels in {time.time() - t0:.1f} s (includes JIT compilation)")

err = np.abs(res["k"] / k_true - 1.0)
print(f"  k    median error {np.median(err):.1%}   90th percentile {np.percentile(err, 90):.1%}")
print(f"  f    median error {np.median(np.abs(res['f'] / params.cest_pools[0].f - 1)):.1%}")
print(f"  at bounds: {int(res.at_bounds.sum())} of {res.at_bounds.size} values")

# ---------------------------------------------------------------------------------------------
# What happens if you ignore the calibration
# ---------------------------------------------------------------------------------------------
wrong = Experiment(tile_params(jax_params, N), events, m0, params.mz_loc)
res_wrong = fit(wrong, spec, Z, n_iter=40)
err_wrong = np.abs(res_wrong["k"] / k_true - 1.0)
print(f"\nsame data, calibration maps ignored (nominal B0/B1/T1 for every voxel):")
print(f"  k    median error {np.median(err_wrong):.1%}   "
      f"{np.median(err_wrong) / np.median(err):.1f}x worse")

# ---------------------------------------------------------------------------------------------
# Memory, and a float32 surprise worth knowing about
# ---------------------------------------------------------------------------------------------
# vmap over every voxel allocates for every voxel at once, which a real slice will not fit on a
# card. max_chunk bounds peak memory independently of how many voxels there are.
#
# Mathematically the result should not depend on it: the voxels are independent, and chunking
# is the same computation in pieces. In float32 it does, by around a percent.

res_chunked = fit(problem, spec, Z, n_iter=40, max_chunk=32)
d = np.abs(res_chunked["k"] - res["k"])
print(f"\nmax_chunk=32 vs one shot: max |dk| = {d.max():.2f} Hz "
      f"({(d / res['k']).max():.1%} relative)")
print("""
  That is not a bug, and it is not the chunking being wrong. Changing the batch size changes the
  kernel XLA compiles, float32 reductions are not associative, and the k/f problem is close
  enough to singular that a ~1e-7 difference in the Jacobian moves the optimiser's path by ~1 %
  in k. The same comparison in float64 agrees to 2e-7 relative.

  What to take from it: float32 fitting of k and f is reproducible to about a percent on this
  kind of problem, and that is a floor set by conditioning, not by the data. If a conclusion
  depends on a difference smaller than that, refit with bmc_jax.precision.configure_x64(True)
  and check it survives.""")

print("""
For several GPUs there is fit_phantom_lm_sharded, which splits the same batch across every
local device.

That is the tour: pools (01), simulate (02), a real .seq (03), one-level fitting and its
failure mode (04), the QUESP ladder (05), QUEST (06), and a slice (07).""")
