"""BMC-JAX -- a differentiable Bloch-McConnell simulator for CEST MRI, built on JAX.

The forward model reformulates the affine Bloch-McConnell ODE in HOMOGENEOUS COORDINATES: the
magnetisation state is augmented with a unity component so the affine system becomes a single
linear operator, evaluated as one matrix exponential per time step. No matrix inversion is
needed anywhere.

Two levels of parallelism carry the cost:
  * `jax.vmap` over the full Z-spectrum -- every saturation offset is simulated at once;
  * `jax.lax.scan` over the time axis within each offset, compiled into one XLA loop.

Because the whole thing is JAX, the Jacobian of a Z-spectrum with respect to any physical
parameter comes from automatic differentiation rather than finite differences, which is what
makes gradient-based fitting and Cramer-Rao bounds cheap.

Subpackages
-----------
simulation : the forward model -- parameters as arrays (SimParams), events, the BMC matrix, the time
             step (two matrix exponentials), `simulate_spectrum`, batched voxels, multi-GPU.
sequences  : how a sequence becomes events: built in code (`cw_saturation`) or read from a Pulseq .seq.
fitting    : bounded Levenberg-Marquardt fitting of selected parameters against measured Z-spectra --
             `FitSpec` declares WHICH parameters are free (`FreeParam`), and nothing about any
             particular agent is hardcoded.
io         : formats only -- the Pulseq-CEST parameter format (Parameters, pools, YAML).
mrf        : MR-fingerprinting dictionaries from the simulator.
settings   : global choices: float precision (`use_x64`) and the matrix exponential (`use_expm`).

Precision: float32 is the default and is adequate for k/f fitting. `settings.use_x64(True)` enables
float64 where a near-singular Jacobian needs it.

Renamed on 2026-10-07 (jax_solver -> simulation, precision -> settings, Slot -> FreeParam, ...); the
README has the old -> new table.
"""
from . import settings                                     # noqa: F401  bmc_jax.settings.use_expm(...)

__version__ = "0.2.0"
