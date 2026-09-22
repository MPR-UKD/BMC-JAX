"""The semi-solid magnetisation-transfer pool."""
from dataclasses import dataclass
from typing import ClassVar

from .cest_pool import CESTPool


@dataclass(frozen=True)
class MTPool(CESTPool):
    """The semi-solid pool: an exchanging pool that also needs a lineshape.

    It inherits from :class:`~bmc_jax.io.parameters.cest_pool.CESTPool` because that is what it
    is -- a pool exchanging with water -- and the single thing it adds is the shape of its
    absorption line.

    Attributes
    ----------
    lineshape
        Either ``"lorentzian"`` or ``"superlorentzian"``. Case-insensitive on input and stored
        lowercase.

    Notes
    -----
    The lineshape is not a cosmetic choice. A semi-solid pool has a transverse relaxation time of
    order 10 microseconds, and a Lorentzian that broad is essentially flat across a +-5 ppm
    saturation window. Flat makes it indistinguishable from a change in the overall signal scale,
    so fitting it fits the scale rather than the pool. A SuperLorentzian carries real shape over
    the same window.

    The failure mode is quiet. Choosing the wrong lineshape does not produce an obviously bad fit;
    it reassigns absorption to whichever other parameter is free, which is how an MT arm can
    appear to measure something while measuring nothing.

    Raises
    ------
    ValueError
        If `lineshape` is not one of :attr:`VALID`. Only those two are implemented in the solver,
        so a typo must fail when the config is read rather than produce a plausible wrong spectrum.

    Examples
    --------
    >>> MTPool(f=0.05, t1=1.0, t2=1e-5, k=30, dw=-2.5, lineshape="SuperLorentzian").lineshape
    'superlorentzian'
    """

    lineshape: str = "superlorentzian"

    VALID: ClassVar[tuple[str, ...]] = ("lorentzian", "superlorentzian")
    """The lineshapes the solver implements."""

    def __post_init__(self, t1, t2):
        """Validate and normalise the lineshape after the inherited checks."""
        super().__post_init__(t1, t2)
        if self.lineshape.lower() not in self.VALID:
            raise ValueError(f"lineshape must be one of {self.VALID}, got {self.lineshape!r}")
        object.__setattr__(self, "lineshape", self.lineshape.lower())
