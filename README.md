# BMC-JAX

Differentiable Bloch-McConnell simulation and CEST parameter fitting in JAX.

```python
from bmc_jax.jax_solver.simulator import simulate_spectrum
from bmc_jax.fitting import FitSpec, Slot, fit_phantom_lm
```

## Install

```bash
pip install -e .                 # core: jax + numpy
pip install -e ".[pulseq]"       # + reading/writing .seq
pip install -e ".[reference]"    # + bmctool, for YAML configs and CPU cross-validation
```

The core deliberately depends on **jax and numpy only**. Anything that reads a file format, or
validates against another implementation, is an optional extra -- so an install can never fail
on a dependency the simulator itself does not use.

## Status

v0.1.0 is the solver and the fitting front-end lifted out of the DLQCest research tree, with
imports made package-relative and nothing else changed. It is byte-equivalent to the copy that
produced the phantom results, and that equivalence is the gate every later change has to pass:
see `tests/test_equivalence.py`.

The layout is being reorganised in steps, each one gated on that test:

| step | change |
|------|--------|
| 1 | `jax_solver/` -> `solvers/`, `solver.py` -> `matrix.py` (it builds the BMC matrix; it is not a fitter) |
| 2 | `fitting.py` -> a package: `spec / problem / residual / optimize / batching / result` |
| 3 | optimiser and batching become arguments rather than four separate entry points |
| 4 | `Problem.stack()` for multi-experiment ladders (QUESP over B1, QUEST over saturation time) |
| 5 | native config dataclasses, so bmctool is optional rather than the only way in |

Old names stay exported at every step until the last one.
