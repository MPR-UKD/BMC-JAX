"""precision.py -- one fp32/fp64 switch for the fit scripts.

The fits default to 32-bit (the FAST path): consumer GPUs run fp64 at ~1/64 rate, and fp64 costs
~2.8x even on the measured BMC workload, so fp64-by-default silently made every run ~3x slower
than it needed to be. Set FIT_X64=1 to force the WHOLE fit path back to 64-bit (the SLOW path) --
worth doing when a fit looks numerically suspect (the matrix-exponential step can be stiff at slow
exchange / high B1).

The benchmarks that must SEE both precisions (benchmark_recovery, crb_calibration,
benchmark_fit_scaling) enable x64 themselves and select dtype per fit, so they are unaffected by
this default -- they are exactly the ones producing the "is fp32 good enough?" evidence.

Call configure_x64() right after `import jax`, BEFORE importing jax.numpy or creating any array
(jax.config must be set before the first array). With x64 disabled JAX canonicalises every
float64 request down to float32, so the rest of the code needs no changes.

    import jax
    from precision import configure_x64
    configure_x64()
    import jax.numpy as jnp

(benchmark_fit_scaling is intentionally exempt: it always enables x64 because it explicitly
times BOTH precisions and needs fp64 available regardless of FIT_X64.)
"""
import os
import jax

_TRUE = ("1", "true", "yes", "on")


def x64_enabled() -> bool:
    """True only if FIT_X64 is set to a truthy value (1/true/yes/on). Default: 32-bit fast path."""
    return os.environ.get("FIT_X64", "0").strip().lower() in _TRUE


def configure_x64() -> bool:
    """Set JAX precision from FIT_X64 (default 32-bit FAST). Returns True if 64-bit is active."""
    on = x64_enabled()
    jax.config.update("jax_enable_x64", on)
    return on
