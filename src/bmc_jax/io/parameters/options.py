"""Simulation settings that are not physics."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    """Numerical and bookkeeping settings.

    Attributes
    ----------
    verbose
        Print progress while simulating.
    reset_init_mag
        Restore the magnetisation to `scale` before each saturation block.
    scale
        Magnetisation level to reset to, as a fraction of equilibrium. Must lie in [0, 1].
    max_pulse_samples
        Number of samples every shaped pulse is padded to.

    Notes
    -----
    `max_pulse_samples` is the setting with consequences. Shaped pulses are padded to a fixed
    length so that every event block has the same static shape, which is what lets the whole
    sequence compile into a single loop. Set it too low and a pulse is truncated; set it too high
    and every block pays the cost of the longest one.

    It also makes cached forward models non-interchangeable: a sequence converted at one value
    cannot be reused at another, and mixing them produces a wrong simulation rather than an error.
    Anywhere such a cache exists, the value it was built at is worth recording and checking.

    `scale` below 1.0 describes a sequence whose magnetisation has not fully recovered when
    saturation begins, which is the usual situation when readouts follow one another closely.

    Raises
    ------
    ValueError
        If `scale` is outside [0, 1] or `max_pulse_samples` is not positive.

    Examples
    --------
    >>> Options(max_pulse_samples=200).scale
    1.0
    """

    verbose: bool = False
    reset_init_mag: bool = True
    scale: float = 1.0
    max_pulse_samples: int = 500

    def __post_init__(self):
        """Validate the reset level and the pulse padding length."""
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"scale must be in [0, 1], got {self.scale}")
        if self.max_pulse_samples <= 0:
            raise ValueError(f"max_pulse_samples must be positive, got {self.max_pulse_samples}")
