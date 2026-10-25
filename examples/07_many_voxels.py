"""Tutorial 07: Fitting many voxels at once

The previous tutorials fitted one to three spectra. An image slice contains thousands of
Z-spectra, and every voxel has its own measured water T1, B0 offset and relative B1. These
calibration values are measured separately and held fixed in the fit, while k and f are fitted
in every voxel. BMC-JAX fits all voxels in one batched call.

This tutorial
  1. simulates a small slice of 16 x 16 voxels with a three-rung QUESP ladder (0.5, 1 and 2 uT),
     with k varying from 80 to 900 1/s across the slice, realistic variations of B0, B1 and T1,
     and 0.3 % noise;
  2. fits k and f in every voxel, with the per-voxel calibration held fixed (tile_params gives
     every voxel its own copy of the parameters);
  3. repeats the fit with the nominal B0, B1 and T1 in every voxel, to show the error caused by
     ignoring the calibration maps;
  4. repeats the fit in chunks of 32 voxels (max_chunk), which limits GPU memory, and shows that
     float32 results then differ by up to about one percent.

Requirements: the core package only.

Run:  python 07_many_voxels.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from bmc_jax.fitting import FitSpec, Experiment, FreeParam, fit
from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state, simulate_spectrum, tile_params
from bmc_jax.sequences import cw_saturation

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
ppm = np.linspace(-6.0, 6.0, 31)
N = 16 * 16                                   # a small "slice"
NOISE = 0.003

# ---------------------------------------------------------------------------------------------
# 1. A simulated slice with varying k and varying calibration
# ---------------------------------------------------------------------------------------------
# tile_params gives every voxel its own copy of the parameters, which are then overwritten per
# voxel. Everything that is not overwritten stays the same in all voxels.

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
            initial_state(jax_params), params.mz_loc)
    for b in ladder_b1]).events

m0 = jnp.stack([initial_state(jax.tree.map(lambda a, i=i: a[i], base))
                for i in range(N)])
Z = np.stack([np.asarray(simulate_spectrum(jax.tree.map(lambda a, i=i: a[i], base), events,
                                           m0[i], return_full_trajectory=False))[:, params.mz_loc]
              for i in range(N)])
Z = Z + rng.normal(0, NOISE, Z.shape)
print(f"slice: {N} voxels x {Z.shape[1]} offsets ({len(ladder_b1)} B1 rungs x {len(ppm)})")

# ---------------------------------------------------------------------------------------------
# 2. Fit with the per-voxel calibration held fixed
# ---------------------------------------------------------------------------------------------
# b0_inhom, rel_b1 and water_r1 are set per voxel and held at their measured values. Using one
# average value for the whole slice instead is a common error: a 6 % error in B1 becomes a larger
# error in the fitted k, because the saturation efficiency depends on the square of the amplitude.

fit_base = tile_params(jax_params, N)._replace(
    b0_inhom=jnp.asarray(b0_true), rel_b1=jnp.asarray(b1_true),
    water_r1=jnp.asarray(1.0 / t1_true))
problem = Experiment(fit_base, events, m0, params.mz_loc)

spec = FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=20000.0, x0=300.0),
                FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0005)])

t0 = time.time()
res = fit(problem, spec, Z, n_iter=40, n_per_block=len(ppm))
print(f"\nfitted {N} voxels in {time.time() - t0:.1f} s (includes JIT compilation)")

err = np.abs(res["k"] / k_true - 1.0)
print(f"  k    median error {np.median(err):.1%}   90th percentile {np.percentile(err, 90):.1%}")
print(f"  f    median error {np.median(np.abs(res['f'] / params.cest_pools[0].f - 1)):.1%}")
print(f"  at bounds: {int(res.at_bounds.sum())} of {res.at_bounds.size} values")

# ---------------------------------------------------------------------------------------------
# 3. The same fit with the calibration ignored
# ---------------------------------------------------------------------------------------------
wrong = Experiment(tile_params(jax_params, N), events, m0, params.mz_loc)
res_wrong = fit(wrong, spec, Z, n_iter=40)
err_wrong = np.abs(res_wrong["k"] / k_true - 1.0)
print(f"\nsame data, calibration maps ignored (nominal B0/B1/T1 for every voxel):")
print(f"  k    median error {np.median(err_wrong):.1%}   "
      f"{np.median(err_wrong) / np.median(err):.1f}x worse")

# ---------------------------------------------------------------------------------------------
# 4. Memory use, and the reproducibility of float32 fits
# ---------------------------------------------------------------------------------------------
# A batch over every voxel allocates memory for every voxel at once, which does not fit on a GPU
# for a real slice. max_chunk limits the peak memory, whatever the number of voxels.
#
# Mathematically, the result does not depend on the chunk size: the voxels are independent, and
# chunking only splits the same computation into pieces. In float32 the results differ slightly.

res_chunked = fit(problem, spec, Z, n_iter=40, max_chunk=32)
d = np.abs(res_chunked["k"] - res["k"])
print(f"\nmax_chunk=32 vs one shot: max |dk| = {d.max():.2f} Hz "
      f"({(d / res['k']).max():.1%} relative)")
print("""
  This is not an error of the chunking. A different batch size makes XLA compile a different
  kernel, float32 sums depend on the order of the terms, and the k/f problem is close enough to
  singular that a difference of about 1e-7 in the Jacobian changes the path of the optimiser and
  moves k by up to about 1 %. The same comparison in float64 agrees to 2e-7 relative.

  Consequence: float32 fits of k and f are reproducible to about one percent for this kind of
  problem. This limit is set by the conditioning of the problem, not by the data. If a conclusion
  depends on a smaller difference, refit with bmc_jax.settings.use_x64(True) and check that it
  still holds.""")

print("""
For several GPUs, fit_batch_sharded splits the same batch across all local devices.

Summary of tutorials 01 to 07: defining a model (01), simulating a spectrum (02), simulating a
Pulseq file (03), fitting from one saturation level and its limits (04), the QUESP ladder (05),
the QUEST ladder (06) and fitting many voxels (07). Tutorials 08 to 11 apply these steps to a
measured phantom, and tutorial 12 computes the smallest uncertainty a protocol allows.""")
