"""Build a saturation sequence in code, with no .seq file and nothing on disk.

`sequences.pulseq` READS Pulseq. This is its missing counterpart: it WRITES the same padded
`SimEvent` blocks directly, so the simulator can be used -- and tested -- by anyone who does not
have a scanner sequence, a Pulseq install, or a pre-built event cache. Without it, pypulseq is
optional only on paper.

The event array is (n_offsets, n_events): the leading axis is what `simulate_spectrum` vmaps
over, and every offset carries its own copy of the block because the RF frequency is what
differs between them.
"""
import numpy as np
import jax.numpy as jnp

from ..simulation.events import SimEvent, EventType

#: Proton gyromagnetic ratio [rad / (s * uT)]. /2pi gives 42.577 Hz/uT.
GAMMA_RAD_PER_S_UT = 267.5153
#: [Hz / T]
GAMMA_HZ_PER_T = 42.5764e6


def hz_per_ppm(b0_T: float) -> float:
    """Chemical-shift scale at this field. 123.2 Hz/ppm at 2.8936 T."""
    return GAMMA_HZ_PER_T * b0_T / 1e6


def cw_saturation(ppm, b1_uT: float, tsat_s: float, b0_T: float = 2.8936,
                  m0_ppm: float | None = None) -> SimEvent:
    """Continuous-wave saturation at each offset, then read.

    One RF block of `tsat_s` at constant amplitude, followed by a zero-duration ADC. This is the
    idealisation a QUESP/QUEST ladder is usually described by: real scanner sequences are pulse
    trains with gaps, and `sequences.pulseq` is what reads those. Use this for tests, for teaching,
    and for any question about the physics that does not depend on the pulse shape.

    Parameters
    ----------
    ppm
        Saturation offsets [ppm], relative to water.
    b1_uT
        Saturation amplitude [uT]. Converted to the Hz the solver expects as
        `b1_uT * GAMMA_RAD_PER_S_UT / 2pi` -- 42.577 Hz per uT.
    tsat_s
        Saturation duration [s]. Sweeping THIS at fixed b1_uT is a QUEST ladder;
        sweeping b1_uT at fixed tsat_s is a QUESP ladder.
    b0_T
        Field strength, used only to turn ppm into the Hz offset of the RF.
    m0_ppm
        If given, prepend one unsaturated reference offset here (e.g. -300 ppm), matching the
        scanner convention of acquiring M0 far off resonance. The returned array then has
        `len(ppm) + 1` rows and the caller must drop or use row 0 accordingly.

    Returns
    -------
    SimEvent
        Batched events, shape (n_offsets, 2), ready for `simulate_spectrum`.
    """
    ppm = np.atleast_1d(np.asarray(ppm, np.float64))
    if m0_ppm is not None:
        ppm = np.concatenate([[float(m0_ppm)], ppm])
    n = ppm.size
    if tsat_s < 0:
        raise ValueError(f"tsat_s must be >= 0, got {tsat_s}")
    if b1_uT < 0:
        raise ValueError(f"b1_uT must be >= 0, got {b1_uT}")

    amp_hz = b1_uT * GAMMA_RAD_PER_S_UT / (2 * np.pi)
    freq_hz = ppm * hz_per_ppm(b0_T)
    zeros = np.zeros(n)

    def pair(sat, adc):
        """Column 0 is the saturation block, column 1 the ADC."""
        return np.stack([np.full(n, sat, np.float64) if np.isscalar(sat) else sat,
                         np.full(n, adc, np.float64) if np.isscalar(adc) else adc], axis=1)

    return SimEvent(
        type=jnp.asarray(pair(EventType.RF_PULSE, EventType.ADC).astype(np.int32)),
        dt=jnp.asarray(pair(tsat_s, 0.0).astype(np.float32)),
        rf_amp=jnp.asarray(pair(amp_hz, 0.0).astype(np.float32)),
        rf_phase=jnp.asarray(pair(0.0, 0.0).astype(np.float32)),
        rf_freq=jnp.asarray(pair(freq_hz, zeros).astype(np.float32)),
        adc_idx=jnp.asarray(pair(-1, 0).astype(np.int32)),
    )
