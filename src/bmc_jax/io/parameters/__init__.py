"""The pool/scanner description a simulation is built from.

One class per file, mirroring the layout of the format's reference implementation, so a class is
where you would look for it. They are plain frozen dataclasses: validation runs once at
construction, and freezing is what makes once enough.

    pool.py         Pool        size fraction, chemical shift, r1, r2
    water_pool.py   WaterPool   the reference pool; adds nothing
    cest_pool.py    CESTPool    + k, the exchange rate to water
    mt_pool.py      MTPool      + lineshape, the semi-solid pool
    system.py       System      field strength and the two calibration maps
    options.py      Options     numerics; max_pulse_samples is the one with teeth
    parameters.py   Parameters  all of the above, plus from_yaml / from_dict
"""
from .cest_pool import CESTPool
from .mt_pool import MTPool
from .options import Options
from .parameters import Parameters
from .pool import Pool, rate_from
from .system import GAMMA_RAD_PER_S_UT, System
from .water_pool import WaterPool

__all__ = ["Pool", "WaterPool", "CESTPool", "MTPool", "System", "Options", "Parameters",
           "GAMMA_RAD_PER_S_UT", "rate_from"]
