"""System -- the scanner: field strength, gyromagnetic ratio, and the two calibration maps."""
from dataclasses import dataclass

#: Proton gyromagnetic ratio [rad / (s * uT)] -- the Pulseq-CEST default.
GAMMA_RAD_PER_S_UT = 267.5153


@dataclass(frozen=True)
class System:
    """Field and calibration.

    b0_inhom and rel_b1 are per-measurement CALIBRATION, not properties of the sample: the actual
    offset and the actual B1 delivered, which differ from nominal. Both belong here rather than
    being baked into the sequence, because both are measured (WASABI, WASSR, double-angle) and
    both are fittable per voxel simply by naming them in a FitSpec.

    DEFAULTS. bmctool requires all four. gamma, b0_inhom and rel_b1 have obvious nominal values,
    so they default here and a config bmctool rejects is accepted. For any config that specifies
    them the two agree exactly -- more forgiving, never different. b0 has no sensible default and
    stays required.
    """
    b0: float
    gamma: float = GAMMA_RAD_PER_S_UT
    b0_inhom: float = 0.0            # [ppm]
    rel_b1: float = 1.0              # dimensionless, 1.0 = nominal

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError(f"b0 must be positive, got {self.b0}")
        if self.rel_b1 <= 0:
            raise ValueError(f"rel_b1 must be positive, got {self.rel_b1}")
