"""The free water pool."""
from dataclasses import dataclass

from .pool import Pool


@dataclass(frozen=True)
class WaterPool(Pool):
    """Free water, the pool everything else is measured against.

    It adds nothing to :class:`~bmc_jax.io.parameters.pool.Pool`, and that is the point. Water is
    the reference, so its chemical shift is zero by definition, and it has no exchange rate of its
    own -- exchange rates belong to the pools that exchange *with* water.

    It exists as a distinct class so that a set of parameters describes itself: ``params.water_pool``
    being a `WaterPool` says which pool is the reference without anyone having to know that it is
    the first one in a list.

    Examples
    --------
    >>> WaterPool(f=1.0, t1=1.3, t2=0.05).dw
    0.0
    """
