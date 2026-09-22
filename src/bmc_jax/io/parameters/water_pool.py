"""WaterPool -- the free water pool."""
from dataclasses import dataclass

from .pool import Pool


@dataclass(frozen=True)
class WaterPool(Pool):
    """Free water. It adds nothing to Pool, and that is the point.

    Water is the reference every other pool is measured against, so its chemical shift is 0 by
    definition and it has no exchange rate TO anything -- the k values live on the pools that
    exchange with it. A separate class rather than a bare Pool because `params.water_pool` being
    a WaterPool is what makes a config self-describing.
    """
