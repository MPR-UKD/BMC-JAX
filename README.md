# BMC-JAX

Differentiable Bloch–McConnell simulation and quantitative CEST fitting, in JAX.

Simulate a Z-spectrum from a pool model, and fit exchange rates and pool sizes back out of
measured data — on GPU, with exact Jacobians from automatic differentiation rather than finite
differences.

```bash
pip install -e .
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

The Bloch–McConnell system is affine, `dM/dt = A M + b`. Augmenting the state with a constant
unity component folds `b` into the matrix, so a time step is **one matrix exponential and never a
matrix inversion**:

```
[ M ]'     [ A  b ] [ M ]
[ 1 ]   =  [ 0  0 ] [ 1 ]
```

Two levels of parallelism carry the cost: `jax.vmap` runs every saturation offset at once, and
`jax.lax.scan` compiles the time evolution within each offset into a single XLA loop.

Because the whole forward model is JAX, the derivative of a Z-spectrum with respect to any
physical parameter comes from autodiff. That is what makes gradient-based fitting and
Cramér–Rao bounds cheap, and it is the difference from a simulator you can only evaluate.

## Install

```bash
pip install -e .                 # core: jax + numpy only
pip install -e ".[pulseq]"       # + pypulseq, to read and write .seq
pip install -e ".[reference]"    # + bmctool, for cross-validation against the CPU reference
pip install -e ".[dev]"          # + pytest
```

The core depends on **jax and numpy and nothing else**. Everything that reads a file format, or
validates against another implementation, is an optional extra, so an install cannot fail on a
dependency the simulator does not use.

## What is in it

```
bmc_jax/
├── simulation/          the forward model
│   ├── params.py            pools -> SimParams, the static array layout JIT needs; initial_state
│   ├── events.py            SimEvent, EventType, padding and casting of event blocks
│   ├── matrix.py            the Bloch–McConnell matrix in homogeneous coordinates
│   ├── expm.py              one time step, with either matrix exponential (see settings)
│   ├── simulate.py          scan over time, vmap over offsets
│   ├── voxels.py            batched over voxels, with per-voxel parameters
│   └── multigpu.py          spread a batch across every local GPU
├── sequences/
│   ├── saturation.py        build a saturation sequence in code, with no file at all
│   └── pulseq.py            or read a Pulseq .seq
├── fitting/
│   ├── experiment.py        what is simulated         (Experiment, Experiment.stack)
│   ├── params.py            which parameters are free (FitSpec, FreeParam)
│   ├── residual.py          what is compared
│   ├── lm.py                bounded Levenberg–Marquardt, usable on its own
│   ├── independent.py       every sample on its own: one spectrum, many voxels, or many GPUs
│   └── result.py            FitResult: values, residual per rung, bounds hit
├── io/                      Pulseq-CEST YAML, as plain frozen dataclasses
├── mrf/                     MR-fingerprinting dictionaries from the simulator
└── settings.py              use_x64 (float precision), use_expm (which matrix exponential)
```

Nothing about any particular agent is hardcoded. A `FitSpec` names fields and pool indices;
which pools exist and what they are called belongs to the model, not to the fitter.

`lm.py` does not import the simulation, so `levenberg_marquardt` is a general bounded least-squares
routine you can use for anything. That boundary is enforced by a test, not by convention.

### Two matrix exponentials

Every time step is one matrix exponential. `bmc_jax.settings.use_expm` picks how it is computed:

```python
import bmc_jax
bmc_jax.settings.use_expm("library")   # default: jax.scipy.linalg.expm (cuBLAS / cuSOLVER on a GPU)
bmc_jax.settings.use_expm("plain")     # the same float32 algorithm, written out: no library call
```

Both give the same answer to float32 rounding. "plain" exists because, on some GPU batch shapes,
the library calls inside `expm` returned NaN or wrong slopes in a vmapped Jacobian while the
residual itself was fine. `simulation/expm.py` has the details.

### Renamed in 2026-10

The layout and several names changed, with no aliases kept: update imports with this table.

| before | now |
|---|---|
| `bmc_jax.jax_solver` | `bmc_jax.simulation`, `bmc_jax.sequences`, `bmc_jax.mrf` |
| `JAXParams`, `convert_params` | `SimParams`, `to_sim_params` |
| `get_initial_magnetization(p, scale=)` | `initial_state(p, m_start=)` |
| `build_bmc_matrix_and_vector`, `solve_step` | `bmc_matrix`, `propagate` |
| `convert_sequence(_batch)` | `events_from_seq(_batch)` |
| `Slot`, `apply_x` | `FreeParam`, `write_params` |
| `lm_minimize` | `levenberg_marquardt` |
| `make_single_residual`, `make_phantom_residual` | `residual_one`, `residual_batch` |
| `fit_voxel`, `fit_phantom_lm(_sharded)`, `fit_phantom_adam` | `fit_one`, `fit_batch(_sharded)`, `fit_batch_adam` |
| `bmc_jax.precision.configure_x64` | `bmc_jax.settings.use_x64` |

## Tutorials

Nine scripts in [`examples/`](examples/), each runnable on its own; 01–07 need no data files, 08 and 09
fit the phantom measurement that ships in `examples/data/`:

| | |
|---|---|
| `01_pools.py` | build water, CEST and MT pools in code and from YAML |
| `02_simulate.py` | pools to Z-spectrum; why MTRasym is not a clean per-pool measure |
| `03_seq_file.py` | write a real `.seq`, read it back, simulate it |
| `04_fit_one.py` | fit k and f from one saturation level — and find where that quietly fails |
| `05_fit_ladder.py` | the QUESP ladder |
| `06_quest.py` | the saturation-time ladder, and QUESP vs QUEST as a cross-check |
| `07_many_voxels.py` | a slice, with per-voxel calibration held fixed |
| `08_quesp.py` | QUESP on real phantom data |
| `09_quest.py` | QUEST on the same phantom |

## Two things worth knowing before you trust a number

**A `k` from one saturation level may not be a measurement of `k`.** Saturation efficiency is
`alpha = w1² / (w1² + k(k + R2s))`, and it is the lever the fit uses to separate an exchange rate
from a pool size. Where `alpha` is near 1 the pool is already saturated and raising `k` changes
nothing; where it is near 0 almost nothing is saturated. Either way the lever is short and noise
fills the gap. On otherwise identical data with 0.2 % noise, the same fit is 3 % wrong at
`k = 500 Hz` and 104 % wrong at `k = 50 Hz` — converged, no bounds hit, healthy residual, both
times. `04_fit_one.py` reproduces this; `05_fit_ladder.py` shows what a ladder recovers and what
it does not.

**float32 fitting is reproducible to about a percent.** Changing the batch size changes the
kernel XLA compiles, float32 reductions are not associative, and a k/f fit is close enough to
singular that a ~1e-7 difference in the Jacobian moves the result by ~1 % in `k`. Measured: 1.1e-2
relative between one shot and `max_chunk=32`, against 1.8e-7 in float64. That is well below the
error noise imposes — on 0.2 % noise, float32 and float64 land 2.147 % and 2.054 % from truth —
so fit in float32 and re-check anything that turns on a smaller difference with
`bmc_jax.settings.use_x64(True)`.

## Tests

```bash
pytest                    # 63 tests, plus the docstring examples
```

| | checks |
|---|---|
| `test_solver.py` | physics invariants of the forward model |
| `test_experiment.py` | stacking experiments, and the padding that makes it safe |
| `test_fitting.py` | simulate, then fit the parameters back |
| `test_config.py` | reading a Pulseq-CEST config |
| `test_structure.py` | the package's own layering rules |

`test_solver.py` asserts what is true of *any* correct Bloch–McConnell implementation — the far
wing returns to 1, a pool-free spectrum is even in ppm, a pool at `+dw` dips only on its own side,
`k = 0` transfers no label, saturation deepens with both power and time, and the dip stays at the
same ppm at 1.5, 2.9 and 7 T. A stored reference spectrum would only check that the code still
does what it did; these check that what it does is right, and they need no data files.

Docstring examples run as tests, so an example that stops being true fails the build rather than
misleading the next reader.

Every test runs from a clean clone, and every one of them runs: no skips, no environment
variable pointing somewhere else, no optional package standing between you and a green suite.
The package itself depends on numpy and jax and nothing more.

The heaviest end-to-end check is not a test but a tutorial. `examples/08_quesp.py` fits a real
phantom measurement that ships with it -- four tubes, a seven-rung saturation ladder -- and takes
a few minutes. Two of those tubes hold the same solution at the same concentration, so their
fitted pool sizes have to agree, which is the one check there that no calibration error can
satisfy by accident.

## License

MIT. See [LICENSE](LICENSE).
