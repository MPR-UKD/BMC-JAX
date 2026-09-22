"""Pool -- the base class every pool inherits: size, shift, and two relaxation rates."""
from dataclasses import InitVar, dataclass


def rate_from(r: float | None, t: float | None, which: str) -> float:
    """Resolve one relaxation rate from either a rate or a time.

    Giving BOTH is an error, not a preference. If r1 and t1 disagree, one of them is a mistake in
    someone's config, and silently choosing a winner hides it. Giving NEITHER is the same error --
    the rate is not determined -- so one boolean covers both cases.
    """
    if (r is None) == (t is None):
        raise ValueError(f"give either {which}1/{which}2 as a rate or a time, not both and not neither")
    v = float(r) if r is not None else 1.0 / float(t)
    if v <= 0:
        raise ValueError(f"{which} must be positive, got {v}")
    return v


@dataclass(frozen=True)
class Pool:
    """A proton pool: size fraction, chemical shift, and its two relaxation rates.

    FROZEN ON PURPOSE. Validation runs once, in __post_init__, and freezing is what makes that
    enough -- an object that cannot be mutated cannot drift out of the state it was checked in.
    bmctool gets the same guarantee by validating in a property setter per field, which costs it
    a getter/setter pair and a __slots__ entry for every attribute; this costs one word.

    A config is built once from a file and handed to the solver. If you need a variant, build a
    new one: `dataclasses.replace(pool, k=200.0)` re-runs every check.
    """
    f: float = 1.0
    dw: float = 0.0
    r1: float | None = None
    r2: float | None = None
    t1: InitVar[float | None] = None
    t2: InitVar[float | None] = None

    def __post_init__(self, t1, t2):
        # object.__setattr__ because the dataclass is frozen; this is the documented way to
        # finish initialising a frozen instance, and it is the ONLY place it happens.
        object.__setattr__(self, "r1", rate_from(self.r1, t1, "r"))
        object.__setattr__(self, "r2", rate_from(self.r2, t2, "r"))
        object.__setattr__(self, "dw", float(self.dw))
        if not 0.0 <= self.f <= 1.0:
            raise ValueError(f"pool fraction f must be in [0, 1], got {self.f}")

    @property
    def T1(self) -> float:      # noqa: N802 -- T1/T2 are the names physicists use
        """Longitudinal relaxation TIME [s]."""
        return 1.0 / self.r1

    @property
    def T2(self) -> float:      # noqa: N802
        """Transverse relaxation TIME [s]."""
        return 1.0 / self.r2
