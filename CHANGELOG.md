# Changelog

## Unreleased

New:
- `fit_joint(stop_lam=, active_bounds=, keep_jacobian=)`: a stopping rule on the damping, parameters
  held at an active bound, and reuse of the Jacobian after a rejected step, for coupled fits. Off by
  default; the default path is unchanged.
- Automatic tests on every push and once a week (GitHub Actions).

Fixed:
- PyYAML is now a dependency. Reading a model configuration (`example_config`, `from_yaml`) needs it,
  and a clean install without it failed.
- `bmc_jax.sequences.pulseq` no longer needs pypulseq to be imported; pypulseq is only needed to read
  or write a `.seq` file.
- Tutorials 08 to 11 use the plain matrix exponential. With the library version the batched Jacobian
  came back NaN on an RTX 3070, every step was rejected, and the fits ended at their start values.

Changed:
- The tutorials, the tutorial overview and the user guide were rewritten in a neutral, explanatory
  style.

## 0.2.0 (2026-10-18)

The package is reorganised into `simulation/`, `sequences/`, `fitting/`, `mrf/` and `settings.py`.

New:
- `fit_joint`: fit samples together, with values shared by all samples, values free in some samples
  only, factors and offsets, one value per experiment of a ladder, and the start magnetisation.
- `B0Spread`: the B0 spread inside a sample, as an average over isochromats.
- `settings.use_expm`: choose the matrix exponential ("library" or "plain").
- The seven-tube phantom scan of 2026-09-27 in `examples/data/`, with tutorials 08 to 11.

### Renamed

The old names were removed, not kept as aliases. Update imports with this table:

| 0.1.0 | 0.2.0 |
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

## 0.1.0

The package before the reorganisation, with the `jax_solver` names. The phantom fits of the paper on
BMC-JAX were run with this version.
