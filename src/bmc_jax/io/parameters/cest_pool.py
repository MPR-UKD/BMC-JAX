"""A pool that exchanges labile protons with water."""
from dataclasses import dataclass

from .pool import Pool


@dataclass(frozen=True)
class CESTPool(Pool):
    """A solute pool whose labile protons exchange with water.

    Saturating the pool at its own resonance and letting exchange carry that saturation into
    water is what makes the pool visible in a water image, which is the whole CEST mechanism.

    Attributes
    ----------
    k
        Exchange rate to water [Hz]. Must not be negative.

    Notes
    -----
    `k` and `f` are the two quantities a quantitative CEST experiment exists to separate, and they
    resist separation: at a single saturation power the measurement constrains their product, the
    exchange flux, not the factors. Recovering them individually needs a ladder of saturation
    powers (QUESP) or of saturation times (QUEST). A `k` quoted from one saturation level is not
    a measurement of `k`.

    Raises
    ------
    ValueError
        If `k` is negative, in addition to the checks inherited from
        :class:`~bmc_jax.io.parameters.pool.Pool`.

    Examples
    --------
    >>> p = CESTPool(f=0.0009, t1=1.0, t2=0.1, k=50, dw=3.5)
    >>> p.k, p.dw
    (50.0, 3.5)
    """

    k: float = 0.0

    def __post_init__(self, t1, t2):
        """Coerce `k` to float and validate it, after the inherited checks.

        The coercion matters more than it looks. Configs routinely write ``k: 50`` rather than
        ``k: 50.0``, and an integer that reaches the solver becomes an integer array, whose
        derivative is zero -- the parameter then cannot be fitted, silently.
        """
        super().__post_init__(t1, t2)
        object.__setattr__(self, "k", float(self.k))
        if self.k < 0:
            raise ValueError(f"exchange rate k must be >= 0, got {self.k}")
