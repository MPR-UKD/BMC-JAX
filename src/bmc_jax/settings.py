"""Global settings: float32 vs float64 (`use_x64`) and which matrix exponential the solver uses
(`use_expm`, see simulation/expm.py). Formerly precision.py (configure_x64), renamed on 2026-10-07.

PRECISION

WHY 32-BIT IS THE DEFAULT
  Consumer GPUs run fp64 at roughly 1/64 of fp32 throughput, and on the measured Bloch-McConnell
  workload fp64 costs about 2.8x even accounting for that. Defaulting to fp64 silently makes
  every run about three times slower than it needs to be, for accuracy that k/f fitting does not
  use: ~4000 sequential matrix steps in fp32 accumulate a random walk of order 1e-4, well below
  the systematic uncertainties of a real measurement.

WHEN TO TURN IT ON
  When a fit looks numerically suspect rather than physically wrong. The matrix-exponential step
  can be stiff at slow exchange and high B1, and a near-singular Jacobian can have fp32 reading
  noise in its columns -- a Cramer-Rao bound computed in fp32 has been seen to differ from the
  fp64 one by 9x on a slow pool. If a result changes when you flip this, it was never a result.

    import jax
    from bmc_jax.settings import use_x64
    use_x64(True)          # BEFORE importing jax.numpy or creating any array
    import jax.numpy as jnp

  JAX requires the setting before the first array exists. With x64 off, JAX canonicalises every
  float64 request down to float32, so nothing else in the code needs to change.

NO ENVIRONMENT VARIABLES
  This used to read FIT_X64 -- a variable named after the research project this code came from,
  documented nowhere a user of the installed package would look. A library whose global numerical
  precision depends on an undocumented environment variable is not shippable: two identical
  scripts give different answers on two machines and nothing in either says why.

  Precision is an argument. If you want it environment-driven, that is your application's choice
  to make, explicitly:

      use_x64(os.environ.get("MY_APP_X64") == "1")
"""
import jax


def use_x64(enable: bool = False) -> bool:
    """Set JAX's global precision. Returns whether 64-bit is now active.

    Must be called before `import jax.numpy` and before any array is created.
    """
    jax.config.update("jax_enable_x64", bool(enable))
    return x64_enabled()


def x64_enabled() -> bool:
    """Whether JAX is currently in 64-bit mode. Reads JAX, not the environment."""
    return bool(jax.config.read("jax_enable_x64"))


_EXPM = {"backend": "library"}
EXPM_BACKENDS = ("library", "plain")


def use_expm(backend: str = "library") -> str:
    """Select the matrix exponential of the solver's time step: "library" (default) or "plain".

    "library" is jax.scipy.linalg.expm, as every published number used. "plain" is the same float32
    algorithm with no linear-algebra library call (simulation/expm.py says why it exists). JAX's
    compilation cache is cleared, so functions compiled before the switch are recompiled with it.
    """
    if backend not in EXPM_BACKENDS:
        raise ValueError(f"use_expm: one of {EXPM_BACKENDS}, not {backend!r}")
    _EXPM["backend"] = backend
    jax.clear_caches()
    return backend


def expm_backend() -> str:
    """The matrix exponential currently selected: "library" or "plain"."""
    return _EXPM["backend"]
