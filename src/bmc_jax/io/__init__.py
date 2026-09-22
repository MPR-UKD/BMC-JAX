"""Getting data in: Pulseq-CEST YAML configs, and Pulseq .seq sequences.

Everything here is about FORMATS, not physics. The solver never imports this package, so a
format that fails to parse can never produce a wrong simulation -- it produces no simulation.
"""
from .config import (CESTPool, MTPool, Options, Parameters, Pool, System, WaterPool, from_yaml,
                     example_config, example_config_names)

__all__ = ["Parameters", "Pool", "WaterPool", "CESTPool", "MTPool", "System", "Options", "from_yaml",
           "example_config", "example_config_names"]
