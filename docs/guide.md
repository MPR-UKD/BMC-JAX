# User guide

This guide covers joint fits of several samples, the B0 spread inside a sample, the choice of
matrix exponential, and the limits of a fitted exchange rate. For a first introduction, see the
[README](../README.md) and the [tutorials](../examples/README.md).

## Joint fits

`fit` solves every sample independently. When samples share parameters, or when a pool exists in
only some of them, use `fit_joint` and declare this in the fit specification:

```python
from bmc_jax.fitting import FitSpec, FreeParam, fit_joint

spec = FitSpec([
    FreeParam("k", "cest_k", 0, 100.0, 5.0, 2000.0, 300.0),                     # one per sample
    FreeParam("f", "cest_f", 0, 0.001, 1e-5, 0.03, 0.005),
    FreeParam("r2f", "water_r2", None, 1.0, 0.5, 1.5, 1.0,                      # one factor on every
              scope="shared", mode="factor"),                                  # sample's measured R2
    FreeParam("m0", "m_start", None, 1.0, 0.1, 1.0, 0.6, scope="shared"),        # the start state
    FreeParam("db0", "b0_inhom", None, 0.01, -0.1, 0.1, 0.0,                     # a B0 drift per rung,
              scope="block", mode="add"),                                      # on top of the B0 map
    FreeParam("f_noe", "cest_f", 1, 0.001, 0.0, 0.03, 0.002, samples=(0, 1)),   # only where it exists
])
res = fit_joint(ladder, spec, Z, n_per_block=len(ppm))
res["r2f"], res["db0"], res.sd, res.free
```

Options of `FreeParam`:

| Option | Values |
|---|---|
| `scope` | `"sample"` (default): one value per sample. `"shared"`: one value for all samples. `"block"`: one value per experiment of a stacked ladder. |
| `mode` | `"value"` (default): replaces the stored value. `"factor"`: multiplies it. `"add"`: shifts it. |
| `samples` | the samples in which the parameter is free. All other samples keep their stored value. |

`"m_start"` is not a field of `SimParams` but the magnetisation from which the sequence starts. When
it is free as a value, the start state is rebuilt from the current pools at every step.

Parameters that are not free keep the values stored in each sample's parameters. The Jacobian is
built per sample, for that sample's own parameters, so a shared parameter adds one column and not a
separate fit. A per-block parameter adds one forward tangent per block. With
`block_jacobian="split"`, the per-block columns are built from a single tangent and split by rung.
The derivative is the same up to float32 rounding, but in a poorly conditioned coupled fit the
different rounding can lead to a different stopping point. The default therefore keeps the earlier
behaviour. Every way of building the Jacobian is checked against `jax.jacfwd` of the full residual in
`tests/test_joint.py`.

Error bars (`res.sd`) and correlations (`res.corr`) are computed from the Jacobian at the solution
and scaled by the misfit. For a specification without shared or partial parameters, `fit_joint`
returns the same values as `fit`. `fit` rejects a specification it cannot honour instead of ignoring
`scope` or `samples`.

## B0 spread inside a sample

`simulate_spectrum` follows a single isochromat. A real voxel or tube contains spins a few Hz apart,
and under a pulsed saturation train a single isochromat shows oscillations that the measurement does
not.

`fit_joint(..., spread=B0Spread(sigma_hz, shape="gauss" | "lorentz", nodes=5))` simulates every
sample as `nodes` isochromats around its own B0 and averages them. The width is the field
`"b0_spread"` (Hz). It can be fitted like any other parameter, usually as one value shared by all
samples, or held at `sigma_hz`. The fitted width depends on the number of nodes and is therefore an
effective width. Keep `nodes` fixed when comparing fits. `simulate_spectrum_spread` computes the same
average outside a fit.

## Matrix exponential

Every time step is one matrix exponential. `bmc_jax.settings.use_expm` selects how it is computed:

```python
import bmc_jax
bmc_jax.settings.use_expm("library")   # default: jax.scipy.linalg.expm (cuBLAS / cuSOLVER on a GPU)
bmc_jax.settings.use_expm("plain")     # the same float32 algorithm, written out without library calls
```

Both give the same result up to float32 rounding. The `"plain"` option exists because, for some GPU
batch shapes, the library calls inside `expm` returned NaN or incorrect derivatives in a vmapped
Jacobian, while the residual itself was correct. Details are given in `simulation/expm.py`.

## Limits of a fitted exchange rate

**An exchange rate fitted from one saturation level is not always a measurement of k.** The
saturation efficiency

```text
alpha = w1^2 / (w1^2 + k (k + R2s))
```

is what allows a fit to separate the exchange rate from the pool size. When `alpha` is close to 1,
the pool is already saturated. When it is close to 0, the pool is hardly saturated. In both cases
noise dominates the difference. On the same data with 0.2 % noise, the same fit is 3 % off at
k = 500 s⁻¹ and 104 % off at k = 50 s⁻¹, although it converges, hits no bound and leaves an
unremarkable residual in both cases. `examples/04_fit_one.py` demonstrates this, and
`examples/05_fit_ladder.py` shows what a ladder of saturation amplitudes recovers.

**float32 fits are reproducible to about one percent.** A different batch size compiles a different
kernel, float32 sums depend on their order, and a k/f fit is close enough to singular that a
difference of 1e-7 in the Jacobian moves k by about 1 %. Measured: 1.1e-2 relative difference
between one batch and `max_chunk=32`, against 1.8e-7 in float64. This is well below the error caused
by noise: with 0.2 % noise, float32 and float64 fits land 2.147 % and 2.054 % from the true value.
Fit in float32, and check results that depend on smaller differences with
`bmc_jax.settings.use_x64(True)`.
