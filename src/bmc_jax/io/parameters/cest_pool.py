"""CESTPool -- a pool that exchanges labile protons with water."""
from dataclasses import dataclass

from .pool import Pool


@dataclass(frozen=True)
class CESTPool(Pool):
    """An exchanging pool: `k` [Hz] to water, `dw` [ppm] from water.

    k and f are the two numbers a CEST experiment exists to separate, and they are hard to
    separate: at one saturation power the observable constrains their PRODUCT, the exchange flux.
    Recovering them individually needs a ladder -- several B1 levels, or several saturation times.
    """
    k: float = 0.0

    def __post_init__(self, t1, t2):
        super().__post_init__(t1, t2)
        if self.k < 0:
            raise ValueError(f"exchange rate k must be >= 0, got {self.k}")
