"""Everything one simulation needs, assembled from a Pulseq-CEST config."""
from dataclasses import dataclass, field
from pathlib import Path

from .cest_pool import CESTPool
from .mt_pool import MTPool
from .options import Options
from .system import System
from .water_pool import WaterPool


@dataclass(frozen=True)
class Parameters:
    """A complete description of what is being simulated: the pools, the scanner, the settings.

    Attributes
    ----------
    water_pool
        The free water pool.
    cest_pools
        Exchanging solute pools, in the order the config lists them. That order is the pool index
        a `FitSpec` slot refers to.
    mt_pool
        The semi-solid pool, or None if the model has none.
    system
        Field strength and calibration.
    options
        Numerical settings.

    Notes
    -----
    The attribute names here are the ones ``convert_params`` reads when it builds the solver's
    array layout. That coupling is one-directional: the solver never imports this package, it
    reads attributes off whatever object it is handed. So this class is a convenience for getting
    a config file in, not a requirement -- the solver's parameter arrays can be built directly,
    which is how the solver tests run without any YAML.

    Examples
    --------
    >>> p = Parameters.from_dict({"water_pool": {"f": 1.0, "t1": 1.3, "t2": 0.05}, "b0": 3})
    >>> p.num_cest_pools, p.mz_loc
    (0, 2)
    """

    water_pool: WaterPool
    cest_pools: list[CESTPool] = field(default_factory=list)
    mt_pool: MTPool | None = None
    system: System | None = None
    options: Options = field(default_factory=Options)

    @property
    def num_cest_pools(self) -> int:
        """How many exchanging solute pools the model has."""
        return len(self.cest_pools)

    @property
    def mz_loc(self) -> int:
        """Index of water's longitudinal magnetisation in the solver's state vector.

        A Z-spectrum is this one element of the simulated state, read at each saturation offset,
        so every caller that plots or fits a Z-spectrum needs this number.

        It lives here rather than being recomputed by callers because it is an index into the
        solver's internal layout. A caller that hardcodes it keeps working right up until the
        layout changes, and then fails silently by reading a different pool's magnetisation.
        """
        return 2 + 2 * len(self.cest_pools)

    @classmethod
    def from_dict(cls, config: dict) -> "Parameters":
        """Build from an already-parsed Pulseq-CEST config.

        Parameters
        ----------
        config
            Mapping with a ``water_pool`` entry, optional ``cest_pool`` and ``mt_pool`` entries,
            and scanner and option values at the top level. The alternative spellings
            ``b0_inhomogeneity``, ``relb1`` and ``max_samples`` are accepted, because configs in
            circulation use both.

        Returns
        -------
        Parameters

        Raises
        ------
        ValueError
            If there is no water pool, or if any pool, scanner or option value fails its own
            validation.
        """
        alias = {"b0_inhomogeneity": "b0_inhom", "relb1": "rel_b1",
                 "max_samples": "max_pulse_samples"}
        cfg = {alias.get(key, key): value for key, value in config.items()}

        if "water_pool" not in cfg:
            raise ValueError("config has no water_pool")

        water = WaterPool(**cfg["water_pool"])
        cest = [CESTPool(**pool) for pool in (cfg.get("cest_pool") or {}).values()]
        mt = MTPool(**cfg["mt_pool"]) if cfg.get("mt_pool") else None

        system_keys = {"b0", "gamma", "b0_inhom", "rel_b1"}
        option_keys = {"verbose", "reset_init_mag", "scale", "max_pulse_samples"}
        system = System(**{k: v for k, v in cfg.items() if k in system_keys})
        options = Options(**{k: v for k, v in cfg.items() if k in option_keys})
        return cls(water, cest, mt, system, options)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Parameters":
        """Read a Pulseq-CEST YAML config from disk.

        Parameters
        ----------
        path
            Path to the ``.yaml`` file.

        Returns
        -------
        Parameters

        Raises
        ------
        ValueError
            If the file parses but does not describe a valid model.
        """
        import yaml

        with Path(path).open() as handle:
            return cls.from_dict(yaml.safe_load(handle))
