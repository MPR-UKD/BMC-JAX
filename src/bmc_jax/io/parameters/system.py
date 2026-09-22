"""Scanner field strength and per-measurement calibration."""
from dataclasses import dataclass

GAMMA_RAD_PER_S_UT = 267.5153
"""Proton gyromagnetic ratio [rad / (s microtesla)], the Pulseq-CEST default."""


@dataclass(frozen=True)
class System:
    """The scanner: field strength, gyromagnetic ratio, and two calibration terms.

    Attributes
    ----------
    b0
        Field strength [T]. Required; there is no sensible default.
    gamma
        Gyromagnetic ratio [rad / (s microtesla)].
    b0_inhom
        Field offset at this location [ppm], relative to nominal.
    rel_b1
        Delivered saturation amplitude as a fraction of nominal. 1.0 means on-calibration.

    Notes
    -----
    `b0_inhom` and `rel_b1` describe the measurement, not the sample: what the field and the pulse
    amplitude actually were, as opposed to what they were asked to be. They belong here rather
    than being folded into the sequence for two reasons. Both are measured independently, by field
    mapping and by a flip-angle calibration; and both can be fitted per voxel simply by naming
    them in a `FitSpec`, which is only possible while they remain explicit parameters.

    `rel_b1` deserves care in quantitative work. Saturation efficiency goes as the square of the
    amplitude, so a small calibration error is roughly doubled in its effect on a fitted exchange
    rate.

    Defaults differ deliberately from the Pulseq-CEST reference implementation, which requires all
    four values. Three of them have unambiguous nominal values and default here, so a config that
    omits them is accepted rather than rejected. For any config that does specify them the two
    implementations agree exactly.

    Raises
    ------
    ValueError
        If `b0` or `rel_b1` is not positive.

    Examples
    --------
    >>> System(b0=3).rel_b1
    1.0
    """

    b0: float
    gamma: float = GAMMA_RAD_PER_S_UT
    b0_inhom: float = 0.0
    rel_b1: float = 1.0

    def __post_init__(self):
        """Validate the field strength and the amplitude scaling."""
        if self.b0 <= 0:
            raise ValueError(f"b0 must be positive, got {self.b0}")
        if self.rel_b1 <= 0:
            raise ValueError(f"rel_b1 must be positive, got {self.rel_b1}")
