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
jax_solver : sequence and parameter adapters, the forward evolution core (`simulate_spectrum`),
             the per-voxel batch variant, and multi-GPU sharding.
fitting    : bounded Levenberg-Marquardt fitting of selected parameters against measured
             Z-spectra -- `FitSpec` declares WHICH parameters are free, the residual is built
             from the forward model, and nothing about any particular agent is hardcoded.

Precision: float32 is the default and is adequate for k/f fitting. `precision.configure_x64()`
enables float64 where a near-singular Jacobian needs it.
"""
__version__ = "0.1.0"
