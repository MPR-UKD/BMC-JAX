"""Parameters -- everything one simulation needs, assembled from a Pulseq-CEST config."""
from dataclasses import dataclass, field
from pathlib import Path

from .cest_pool import CESTPool
from .mt_pool import MTPool
from .options import Options
from .system import System
from .water_pool import WaterPool


@dataclass(frozen=True)
class Parameters:
    """The attribute names here are the ones `convert_params` reads.

    That coupling is deliberate and one-directional: the solver never imports this package, it
    just reads attributes off whatever it is handed. So these classes are a CONVENIENCE for
    getting a config in, not a requirement -- JAXParams can be built by hand, which is how the
    solver tests run with no YAML anywhere.
    """
    water_pool: WaterPool
    cest_pools: list[CESTPool] = field(default_factory=list)
    mt_pool: MTPool | None = None
    system: System | None = None
    options: Options = field(default_factory=Options)

    @property
    def num_cest_pools(self) -> int:
        return len(self.cest_pools)

    @property
    def mz_loc(self) -> int:
        """Index of water Mz in the solver's state vector: 2 + 2 * n_cest_pools.

        Here rather than recomputed by every caller. It is an index into the solver's internal
        layout, and a caller that hardcodes it is a caller that breaks silently if the layout
        ever changes. The tutorial this code came from had it as a module-level constant.
        """
        return 2 + 2 * len(self.cest_pools)

    @classmethod
    def from_dict(cls, config: dict) -> "Parameters":
        """Build from a parsed Pulseq-CEST config.

        `b0_inhomogeneity`, `relb1` and `max_samples` are accepted as aliases, because configs in
        the wild use both spellings.
        """
        alias = {"b0_inhomogeneity": "b0_inhom", "relb1": "rel_b1",
                 "max_samples": "max_pulse_samples"}
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
