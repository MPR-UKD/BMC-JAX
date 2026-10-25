# Tests

```bash
pip install -e ".[dev]"
pytest                    # 123 tests, the docstring examples included
```

| file | checks |
|---|---|
| `test_solver.py` | physics invariants of the forward model |
| `test_experiment.py` | stacking experiments, and the padding that makes it safe |
| `test_fitting.py` | simulate, then fit the parameters back |
| `test_joint.py` | the same with shared values and a pool in some samples only; a plain spec gives `fit`'s numbers; the B0 spread; the assembled Jacobian against `jax.jacfwd` for every kind of value |
| `test_config.py` | reading a Pulseq-CEST config |
| `test_design.py` | the Cramer-Rao bound (scaling with the noise and with the amount of data) and protocol optimisation |
| `test_structure.py` | the package's own layering rules |

`test_solver.py` checks what is true of any correct Bloch–McConnell implementation: the far wing
returns to 1, a spectrum without pools is even in ppm, a pool at `+dw` dips only on its own side,
`k = 0` transfers no label, saturation deepens with both power and time, and the dip stays at the
same ppm at 1.5, 2.9 and 7 T. A stored reference spectrum would only check that the code still does
what it did. These check that what it does is right, and they need no data files.

The docstring examples run as tests, so an example that stops being true fails the suite.

Every test runs from a clean clone, with no skips, no environment variables and no optional
packages. The package itself depends on numpy, jax and pyyaml only.

The Jacobian check in `test_joint.py` runs at full matmul precision. On a GPU, float32 matrix
products use TF32 by default, and the two Jacobians, built in a different order, would then differ
by about 2e-4 for no fault of the code.

The heaviest end-to-end check is a tutorial, not a test. `examples/08_quesp.py` fits the real
phantom measurement that ships with it (seven tubes, an eight-rung saturation ladder) in a few
minutes. Its tubes come in pairs with a known ratio of pool sizes (2 for GAG at two concentrations,
1 for one solution at two pH), and no calibration error can meet that check by accident.
