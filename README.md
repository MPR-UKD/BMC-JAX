# BMC-JAX

Differentiable Bloch–McConnell simulation and quantitative CEST fitting, in JAX.

BMC-JAX simulates a Z-spectrum from a pool model and a Pulseq sequence, and fits exchange rates and
pool sizes back out of measured data. It runs on a GPU, and the Jacobian of the fit comes from
automatic differentiation instead of finite differences.

![Data flow for one Z-spectrum](docs/figures/dataflow.png)

*One Z-spectrum: the magnetisation is propagated through the sequence events one after the other
(`jax.lax.scan`), at all frequency offsets at once (`jax.vmap`). The same graph gives the Jacobian.*

## Install

```bash
git clone https://github.com/MPR-UKD/BMC-JAX.git
cd BMC-JAX
pip install -e .                 # core: jax + numpy only
pip install -e ".[pulseq]"       # + pypulseq, to read and write .seq files
pip install -e ".[reference]"    # + bmctool, to compare against the CPU reference
pip install -e ".[dev]"          # + pytest
```

## Quickstart

```python
import numpy as np
from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state
from bmc_jax.sequences import cw_saturation
from bmc_jax.fitting import FitSpec, FreeParam, Experiment, fit

params = example_config("amide")          # or from_yaml("your_model.yaml")
jp  = to_sim_params(params)
m0  = initial_state(jp)
ppm = np.linspace(-6, 6, 61)

# a QUESP ladder: the same pools saturated at four powers, fitted as one problem
ladder = Experiment.stack([
    Experiment(jp, cw_saturation(ppm, b1_uT=b, tsat_s=2.0, b0_T=params.system.b0), m0, params.mz_loc)
    for b in (0.5, 1.0, 2.0, 4.0)])

z = np.asarray(ladder.simulate())

spec = FitSpec([FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=5000.0, x0=300.0),
                FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0005)])
res = fit(ladder, spec, z[None, :], n_per_block=len(ppm))
print(res.summary())
```

```
sample             k             f        rms
---------------------------------------------
     0       49.604    0.00090781    3.00e-05      (truth: k = 50, f = 0.0009009)
```

## How it works

The Bloch–McConnell equations are affine, `dM/dt = A M + b`. Adding a constant 1 to the state puts
`b` into the matrix, so every time step is one matrix exponential and no matrix inversion:

```
[ M ]'     [ A  b ] [ M ]
[ 1 ]   =  [ 0  0 ] [ 1 ]
```

The same kernel is then wrapped in more and more axes: the sequence events, the frequency offsets,
the voxels and the GPUs. Every level stays differentiable.

![Batching axes around the kernel](docs/figures/batching_axes.png)

## What is in it

| folder | what it does |
|---|---|
| `simulation/` | the forward model: pools, the matrix, one time step, a spectrum, many voxels, the B0 spread |
| `sequences/` | a saturation sequence built in code, or read from a Pulseq `.seq` file |
| `fitting/` | `fit` (every sample alone) and `fit_joint` (samples with shared values), Levenberg–Marquardt |
| `io/` | Pulseq-CEST YAML configs |
| `mrf/` | MR fingerprinting dictionaries |
| `settings.py` | float32 or float64, and which matrix exponential |

## Tutorials

Eleven scripts in [`examples/`](examples/). Tutorials 01 to 07 build everything in code. Tutorials
08 to 11 fit a real seven-tube phantom measurement that ships in [`examples/data/`](examples/data/).
The list is in [`examples/README.md`](examples/README.md).

## More

- [`docs/guide.md`](docs/guide.md): fitting samples together, the B0 spread, the two matrix
  exponentials, and what to know before you trust a number.
- [`tests/README.md`](tests/README.md): what the 112 tests check. Run them with `pytest`.
- [`CHANGELOG.md`](CHANGELOG.md): what changed between versions, with the table of renamed functions
  for code written for 0.1.0.

## License

MIT. See [LICENSE](LICENSE).
