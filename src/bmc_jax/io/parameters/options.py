"""Options -- simulation settings that are not physics."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    """Numerics and bookkeeping.

    max_pulse_samples is the one with teeth: shaped pulses are padded to this many samples so
    every event block has the same static shape and JIT can compile one loop. Too small and a
    pulse is truncated; too large and every block pays for the longest one. A cached forward model
    built at one value cannot be reused at another, which is worth a guard wherever caches exist.
    """
    verbose: bool = False
    reset_init_mag: bool = True
    scale: float = 1.0
    max_pulse_samples: int = 500

    def __post_init__(self):
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"scale must be in [0, 1], got {self.scale}")
        if self.max_pulse_samples <= 0:
            raise ValueError(f"max_pulse_samples must be positive, got {self.max_pulse_samples}")
