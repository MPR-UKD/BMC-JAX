"""The base class every pool inherits."""
from dataclasses import InitVar, dataclass


def rate_from(r: float | None, t: float | None, which: str) -> float:
    """Resolve a relaxation rate given either the rate or the corresponding time.

    Parameters
    ----------
    r
        The rate [Hz], or None.
    t
        The time [s], or None. Exactly one of `r` and `t` must be given.
    which
        Name used in error messages, ``"r"`` for r1/r2.

    Returns
    -------
    float
        The rate, ``r`` if given, otherwise ``1 / t``.

    Raises
    ------
    ValueError
        If both are given or neither is, or if the result is not positive. Both are the same
        mistake -- the rate is not determined -- so one check covers them. Accepting both and
        preferring one would hide a config in which r1 and t1 disagree.
    """
    if (r is None) == (t is None):
        raise ValueError(f"give either {which}1/{which}2 as a rate or a time, not both and not neither")
    value = float(r) if r is not None else 1.0 / float(t)
    if value <= 0:
        raise ValueError(f"{which} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Pool:
    """A proton pool: how much of it there is, where it resonates, and how it relaxes.

    Relaxation may be given as rates (``r1``, ``r2``) or as times (``t1``, ``t2``), whichever the
    config uses. Times are init-only: after construction the object carries rates and nothing
    else, so there are never two fields that can disagree. The times are still readable through
    the :attr:`T1` and :attr:`T2` properties.

    Instances are immutable. Every value is checked once, at construction, and freezing is what
    makes once enough: an object that cannot be modified cannot drift into a state that was never
    checked. To make a variant, build a new one with :func:`dataclasses.replace`, which re-runs
    every check.

    Attributes
    ----------
    f
        Relative pool size, as a fraction of the total proton pool. Must lie in [0, 1].
    dw
        Chemical shift relative to water [ppm]. Zero for water itself.
    r1
        Longitudinal relaxation rate [Hz].
    r2
        Transverse relaxation rate [Hz].

    Raises
    ------
    ValueError
        If `f` is outside [0, 1], or if the relaxation arguments are ambiguous or non-positive.

    Examples
    --------
    >>> Pool(f=1.0, t1=1.3, t2=0.05).r1
    0.7692307692307692
    >>> Pool(f=1.0, r1=0.77, r2=20.0).T1
    1.2987012987012987
    """

    f: float = 1.0
    dw: float = 0.0
    r1: float | None = None
    r2: float | None = None
    t1: InitVar[float | None] = None
    t2: InitVar[float | None] = None

    def __post_init__(self, t1, t2):
        """Resolve rates from times and validate. Called automatically after construction.

        Uses ``object.__setattr__`` because the class is frozen. That is the documented way to
        finish initialising a frozen dataclass, and this is the only place the package does it.
        """
        object.__setattr__(self, "r1", rate_from(self.r1, t1, "r"))
        object.__setattr__(self, "r2", rate_from(self.r2, t2, "r"))
        object.__setattr__(self, "dw", float(self.dw))
        object.__setattr__(self, "f", float(self.f))
        if not 0.0 <= self.f <= 1.0:
            raise ValueError(f"pool fraction f must be in [0, 1], got {self.f}")

    @property
    def T1(self) -> float:
        """Longitudinal relaxation time [s], the reciprocal of :attr:`r1`.

        Capitalised because T1 and T2 are how these quantities are written in the literature and
        on every scanner console. The lowercase ``t1`` and ``t2`` names are the constructor
        arguments; they do not survive construction.
        """
        return 1.0 / self.r1

    @property
    def T2(self) -> float:
        """Transverse relaxation time [s], the reciprocal of :attr:`r2`."""
        return 1.0 / self.r2
