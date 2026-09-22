"""MTPool -- the semi-solid magnetisation-transfer pool."""
from dataclasses import dataclass
from typing import ClassVar

from .cest_pool import CESTPool


@dataclass(frozen=True)
class MTPool(CESTPool):
    """An exchanging pool that additionally needs a LINESHAPE.

    It inherits from CESTPool because that is what it is: a pool exchanging with water. The one
    thing it adds is the shape of its absorption line, and that choice is not cosmetic.

    A semi-solid pool has T2 of order 10 us. A Lorentzian that broad is essentially FLAT across a
    +-5 ppm window, which makes it indistinguishable from a change in the overall signal scale --
    fit one and you are fitting the scale, not the pool. A SuperLorentzian carries real shape over
    the same window. Choosing the wrong one does not fail loudly; it quietly reassigns absorption
    to whatever else is free, which is how an MT arm can look like it is measuring something.
    """
    lineshape: str = "superlorentzian"

    #: Only these are implemented in the solver. A typo must fail at load rather than produce a
    #: plausible wrong spectrum.
    VALID: ClassVar[tuple[str, ...]] = ("lorentzian", "superlorentzian")

    def __post_init__(self, t1, t2):
        super().__post_init__(t1, t2)
        if self.lineshape.lower() not in self.VALID:
            raise ValueError(f"lineshape must be one of {self.VALID}, got {self.lineshape!r}")
        object.__setattr__(self, "lineshape", self.lineshape.lower())
