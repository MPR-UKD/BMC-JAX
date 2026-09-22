"""Reading a Pulseq-CEST YAML config.

The classes live in `bmc_jax.io.parameters`, one per file. This module is only the entry point
and the configs shipped with the package.

WHY THIS EXISTS
  `convert_params` turns a pool description into the JAXParams the solver wants. It is duck-typed
  -- it reads attributes and never checks a class -- but the only thing that BUILT a suitable
  object was bmctool, so reading a config meant depending on it. The package can now go from a
  .yaml to a simulated Z-spectrum with nothing installed but numpy, jax and pyyaml.

  bmctool is still supported and still useful: its CPU solver is the independent implementation
  the JAX engine is validated against, and replacing that with our own code would destroy the
  validation rather than improve it. It is simply no longer the only way in.

THE FORMAT -- https://pulseq-cest.github.io
  Pools carry f, t1, t2 and, except for water, k and dw. Rates or times, never both:

      water_pool: { f: 1.0, t1: 1.0, t2: 0.040 }
      cest_pool:
        amide: { f: 0.0009, t1: 1.0, t2: 0.1, k: 50, dw: 3.5 }
      mt_pool: { f: 0.05, t1: 1.0, t2: 1.0e-05, k: 30, dw: -2.5, lineshape: SuperLorentzian }
      b0: 3
      gamma: 267.5153
      b0_inhom: 0.0
      rel_b1: 1.0
"""
from pathlib import Path

from .parameters import (GAMMA_RAD_PER_S_UT, CESTPool, MTPool, Options, Parameters, Pool,
                         System, WaterPool)

CONFIG_DIR = Path(__file__).parent / "configs"


def from_yaml(path: str | Path) -> Parameters:
    """Read a Pulseq-CEST YAML config. The usual entry point."""
    return Parameters.from_yaml(path)


def example_config(name: str = "amide") -> Parameters:
    """Load one of the configs shipped with the package, so tutorials need no files.

    >>> p = example_config("amide_noe_mt")
    >>> p.num_cest_pools, p.mt_pool.lineshape
    (2, 'superlorentzian')
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no example config {name!r}; available: {example_config_names()}")
    return from_yaml(path)


def example_config_names() -> list[str]:
    """The configs shipped with the package. They cover every branch the format has."""
    return sorted(q.stem for q in CONFIG_DIR.glob("*.yaml"))
