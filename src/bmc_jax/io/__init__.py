"""Getting data in: Pulseq-CEST YAML configs, and Pulseq .seq sequences.

Everything here is about FORMATS, not physics. The solver never imports this package -- checked
by tests/test_structure.py -- so a config that fails to parse can never produce a wrong
simulation. It produces no simulation.
"""
from .config import CONFIG_DIR, example_config, example_config_names, from_yaml
from .parameters import (GAMMA_RAD_PER_S_UT, CESTPool, MTPool, Options, Parameters, Pool,
                         System, WaterPool)

__all__ = ["Parameters", "Pool", "WaterPool", "CESTPool", "MTPool", "System", "Options",
           "from_yaml", "example_config", "example_config_names", "CONFIG_DIR",
           "GAMMA_RAD_PER_S_UT"]
