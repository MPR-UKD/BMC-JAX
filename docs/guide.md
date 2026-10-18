# Guide

## Fitting samples together

`fit` solves every sample alone. When the samples have something in common, or a pool exists in only
some of them, use `fit_joint` and say so in the spec:

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

- `scope`: one value per sample (the default), one for all samples (`"shared"`), or one per
  experiment of a stacked ladder (`"block"`).
- `mode`: replace the carried value (the default), multiply it (`"factor"`) or shift it (`"add"`).
- `"m_start"` is not a SimParams field but the magnetisation the sequence starts from. When it is
  free as a value, the start state is rebuilt from the current pools at every step.

Where a value is not free, the sample keeps what its params carry. The Jacobian is built per sample,
for that sample's own parameters, so a shared value costs one more column and not another fit. A
per-block value costs one forward tangent per block. `block_jacobian="split"` builds it from one
tangent and splits it by rung. That is the same derivative up to float32 rounding, but it can still
send a badly conditioned coupled fit to a different stall, so the default keeps the earlier numbers.
Every way of building the Jacobian is checked against `jax.jacfwd` of the whole residual in
`tests/test_joint.py`.

Error bars (`res.sd`) and correlations (`res.corr`) come from the Jacobian at the solution, scaled by
the misfit. With a plain spec, `fit_joint` gives the same numbers as `fit`. `fit` refuses a spec it
cannot honour instead of ignoring `scope` or `samples`.

## The B0 spread inside a sample

`simulate_spectrum` follows one isochromat. A real voxel or tube holds spins a few Hz apart, and under
a pulsed train a single isochromat rings where the measurement does not.

`fit_joint(..., spread=B0Spread(sigma_hz, shape="gauss" | "lorentz", nodes=5))` simulates every
sample as `nodes` isochromats around its own B0 and averages them. The width is the field
`"b0_spread"` (Hz). It is fitted like any other value, usually one shared by all samples, or held at
`sigma_hz`. The fitted width depends on the number of nodes (it is an effective width), so keep
`nodes` fixed when you compare fits. `simulate_spectrum_spread` is the same average outside a fit.

## Two matrix exponentials

Every time step is one matrix exponential. `bmc_jax.settings.use_expm` picks how it is computed:

```python
import bmc_jax
bmc_jax.settings.use_expm("library")   # default: jax.scipy.linalg.expm (cuBLAS / cuSOLVER on a GPU)
bmc_jax.settings.use_expm("plain")     # the same float32 algorithm, written out: no library call
```

Both give the same answer to float32 rounding. "plain" exists because, for some GPU batch shapes,
the library calls inside `expm` returned NaN or wrong slopes in a vmapped Jacobian while the
residual itself was fine. `simulation/expm.py` has the details.

## Before you trust a number

**A `k` from one saturation level may not be a measurement of `k`.** The saturation efficiency
`alpha = w1² / (w1² + k(k + R2s))` is what lets the fit separate an exchange rate from a pool size.
Where `alpha` is near 1 the pool is already saturated, and where it is near 0 almost nothing is
saturated. In both cases noise fills the gap. On the same data with 0.2 % noise, the same fit is 3 %
wrong at `k = 500 Hz` and 104 % wrong at `k = 50 Hz`, converged, with no bounds hit and a healthy
residual both times. `examples/04_fit_one.py` shows this, and `examples/05_fit_ladder.py` shows what
a ladder recovers.

**float32 fitting is reproducible to about a percent.** A different batch size compiles a different
kernel, float32 sums depend on their order, and a k/f fit is close enough to singular that a 1e-7
difference in the Jacobian moves `k` by about 1 %. Measured: 1.1e-2 relative between one shot and
`max_chunk=32`, against 1.8e-7 in float64. That is well below the error from noise (on 0.2 % noise,
float32 and float64 land 2.147 % and 2.054 % from the truth). So fit in float32, and re-check
anything that depends on a smaller difference with `bmc_jax.settings.use_x64(True)`.
