"""Read a Pulseq-CEST YAML config into plain Python objects.

WHY THIS EXISTS
  `convert_params` turns a pool description into the JAXParams the solver wants. It is duck-typed
  -- it reads attributes and never checks a class -- but until now the only thing that BUILT such
  an object was bmctool, so reading a config meant depending on bmctool. These dataclasses close
  that gap: the core package can now go from a .yaml to a simulated Z-spectrum with nothing
  installed but numpy, jax and pyyaml.

  bmctool is still supported and still useful (its CPU solver is the independent reference the
  JAX engine is validated against). It is simply no longer the only way in.

THE FORMAT
  https://pulseq-cest.github.io -- pools carry f, t1, t2 and, except for water, k and dw:

      water_pool: { f: 1.0, t1: 1.0, t2: 0.040 }
      cest_pool:
        amide: { f: 0.0009, t1: 1.0, t2: 0.1, k: 50, dw: 3.5 }
      mt_pool: { f: 0.05, t1: 1.0, t2: 1.0e-05, k: 30, dw: -2.5, lineshape: SuperLorentzian }
      b0: 3
      gamma: 267.5153
      b0_inhom: 0.0
      rel_b1: 1.0

  Rates or times, never both: give r1 or t1, r2 or t2. Supplying both is an error rather than a
  silent preference, because the two disagreeing is a mistake worth hearing about.

DELIBERATE DIFFERENCE FROM BMCTOOL
  bmctool's System requires b0, gamma, b0_inhom and rel_b1 to all be present and raises if any is
  missing. The defaults below (gamma 267.5153, b0_inhom 0, rel_b1 1) accept configs it would
  reject. For any config that does specify them the two agree exactly; this is only more
  forgiving, never different.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path

#: Proton gyromagnetic ratio [rad / (s * uT)] -- the Pulseq-CEST default.
GAMMA_RAD_PER_S_UT = 267.5153


def _rate(r: float | None, t: float | None, which: str) -> float:
    """One of r (a rate) or t (a time), never both and never neither."""
    if (r is None) == (t is None):
        raise ValueError(f"give either {which}1/{which}2 as a rate or a time, not both and not neither")
    v = float(r) if r is not None else 1.0 / float(t)
    if v <= 0:
        raise ValueError(f"{which} must be positive, got {v}")
    return v


@dataclass
class Pool:
    """A proton pool: size fraction, chemical shift, and its two relaxation rates."""
    f: float = 1.0
    dw: float = 0.0
    r1: float | None = None
    r2: float | None = None
    t1: InitVar[float | None] = None
    t2: InitVar[float | None] = None

    def __post_init__(self, t1, t2):
        self.r1 = _rate(self.r1, t1, "r")
        self.r2 = _rate(self.r2, t2, "r")
        self.dw = float(self.dw)
        if not 0.0 <= self.f <= 1.0:
            raise ValueError(f"pool fraction f must be in [0, 1], got {self.f}")

    @property
    def T1(self) -> float:      # noqa: N802 -- T1/T2 are the names physicists use
        return 1.0 / self.r1

    @property
    def T2(self) -> float:      # noqa: N802
        return 1.0 / self.r2


@dataclass
class WaterPool(Pool):
    """The free water pool. It is the reference, so dw is 0 and there is no exchange rate."""


@dataclass
class CESTPool(Pool):
    """An exchanging pool: `k` [Hz] to water, `dw` [ppm] from water."""
    k: float = 0.0

    def __post_init__(self, t1, t2):
        super().__post_init__(t1, t2)
        if self.k < 0:
            raise ValueError(f"exchange rate k must be >= 0, got {self.k}")


@dataclass
class MTPool(CESTPool):
    """The semi-solid pool. Unlike a CEST pool it needs a LINESHAPE.

    That choice is not cosmetic. A Lorentzian at the T2 an MT pool actually has (~10 us) is
    essentially flat across a +-5 ppm window, so it cannot be told apart from a change in the
    overall signal scale; a SuperLorentzian carries real shape over the same window. Fitting one
    when the data wants the other quietly reassigns absorption to whatever else is free.
    """
    lineshape: str = "superlorentzian"
    VALID = ("lorentzian", "superlorentzian")

    def __post_init__(self, t1, t2):
        super().__post_init__(t1, t2)
        if self.lineshape.lower() not in self.VALID:
            raise ValueError(f"lineshape must be one of {self.VALID}, got {self.lineshape!r}")
        self.lineshape = self.lineshape.lower()


@dataclass
class System:
    """The scanner: field, gyromagnetic ratio, and the two calibration maps."""
    b0: float
    gamma: float = GAMMA_RAD_PER_S_UT
    b0_inhom: float = 0.0            # [ppm]
    rel_b1: float = 1.0              # dimensionless, 1.0 = nominal

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError(f"b0 must be positive, got {self.b0}")
        if self.rel_b1 <= 0:
            raise ValueError(f"rel_b1 must be positive, got {self.rel_b1}")


@dataclass
class Options:
    """Simulation settings that are not physics."""
    verbose: bool = False
    reset_init_mag: bool = True
    scale: float = 1.0
    max_pulse_samples: int = 500

    def __post_init__(self):
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"scale must be in [0, 1], got {self.scale}")
        if self.max_pulse_samples <= 0:
            raise ValueError(f"max_pulse_samples must be positive, got {self.max_pulse_samples}")


@dataclass
class Parameters:
    """Everything a simulation needs, with the same attribute names `convert_params` reads."""
    water_pool: WaterPool
    cest_pools: list[CESTPool] = field(default_factory=list)
    mt_pool: MTPool | None = None
    system: System = None
    options: Options = field(default_factory=Options)

    @property
    def num_cest_pools(self) -> int:
        return len(self.cest_pools)

    @property
    def mz_loc(self) -> int:
        """Index of water Mz in the state vector: 2 + 2 * n_cest_pools.

        Worth having here rather than recomputed by every caller -- it is an index into the
        solver's internal layout, and callers that hardcode it are the reason a change to the
        layout would be silent.
        """
        return 2 + 2 * len(self.cest_pools)

    @classmethod
    def from_dict(cls, config: dict) -> "Parameters":
        """Build from a parsed Pulseq-CEST config.

        `b0_inhomogeneity`, `relb1` and `max_samples` are accepted as aliases, because configs in
        the wild use both spellings.
        """
        alias = {"b0_inhomogeneity": "b0_inhom", "relb1": "rel_b1", "max_samples": "max_pulse_samples"}
        cfg = {alias.get(k, k): v for k, v in config.items()}

        if "water_pool" not in cfg:
            raise ValueError("config has no water_pool")
        water = WaterPool(**cfg["water_pool"])
        cest = [CESTPool(**p) for p in (cfg.get("cest_pool") or {}).values()]
        mt = MTPool(**cfg["mt_pool"]) if cfg.get("mt_pool") else None

        sys_keys = {"b0", "gamma", "b0_inhom", "rel_b1"}
        opt_keys = {"verbose", "reset_init_mag", "scale", "max_pulse_samples"}
        system = System(**{k: v for k, v in cfg.items() if k in sys_keys})
        options = Options(**{k: v for k, v in cfg.items() if k in opt_keys})
        return cls(water, cest, mt, system, options)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Parameters":
        import yaml
        with Path(path).open() as fh:
            return cls.from_dict(yaml.safe_load(fh))


def from_yaml(path: str | Path) -> Parameters:
    """Read a Pulseq-CEST YAML config. The usual entry point."""
    return Parameters.from_yaml(path)


CONFIG_DIR = Path(__file__).parent / "configs"


def example_config(name: str = "amide") -> Parameters:
    """Load one of the configs shipped with the package. No files of your own needed.

    >>> p = example_config("amide_noe_mt")
    >>> p.num_cest_pools, p.mt_pool.lineshape
    (2, 'superlorentzian')
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        have = sorted(q.stem for q in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no example config {name!r}; available: {have}")
    return from_yaml(path)


def example_config_names() -> list[str]:
    return sorted(q.stem for q in CONFIG_DIR.glob("*.yaml"))
