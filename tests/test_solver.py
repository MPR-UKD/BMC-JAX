"""Physics invariants of the forward model. Self-contained -- no files, no cache, no reference.

WHY NOT A GOLDEN FILE
  A stored reference spectrum tests that the code still does what it did, which is worth having
  (test_equivalence_solver.py does exactly that, against the research tree). It does not test
  that what it does is RIGHT: a golden file generated from a broken solver passes forever.

  These assertions are statements about Bloch-McConnell that hold for any correct implementation
  at any field, so they would have caught a sign error, a ppm/Hz mix-up or a swapped pool index
  on the day it was written. They need nothing but the package itself, which is the point: a
  general tool has to be testable by someone who has never seen this phantom.

TOLERANCES
  Everything runs in float32. The symmetry check below is held to 2e-3 rather than 0 because
  ~4000 sequential matrix steps accumulate a random walk of order 1e-4 -- the same floor that
  sets how precisely the phantom fits can be read. A tighter bound here would be testing the
  arithmetic, not the physics.
"""
import numpy as np
import pytest

from bmc_jax.jax_solver import (JAXParams, cw_saturation, get_initial_magnetization,
                                hz_per_ppm, simulate_spectrum)

B0_T = 2.8936
WATER_Z = 4          # index of water Mz for a one-CEST-pool model: 2 * (1 + n_pools)
DW = 3.5             # amide-like pool offset [ppm]


def params(f=0.005, k=300.0, dw=DW, t2_pool_s=0.01):
    """One water pool, one CEST pool, no MT. Values are ordinary amide-CEST numbers."""
    s = np.float32
    return JAXParams(
        water_r1=s(1 / 3.0), water_r2=s(1 / 0.6), water_f=s(1.0),
        cest_k=np.float32([k]), cest_f=np.float32([f]), cest_dw=np.float32([dw]),
        cest_r1=np.float32([1.0]), cest_r2=np.float32([1.0 / t2_pool_s]),
        mt_k=s(0), mt_f=s(0), mt_dw=s(0), mt_r1=s(0), mt_r2=s(1),
        mt_lineshape=np.int32(0),
        b0=s(B0_T), gamma=s(267.5153), b0_inhom=s(0), rel_b1=s(1.0))


def z_spectrum(p, ppm, b1_uT=1.0, tsat_s=2.0):
    ev = cw_saturation(ppm, b1_uT=b1_uT, tsat_s=tsat_s, b0_T=B0_T)
    return np.asarray(simulate_spectrum(p, ev, get_initial_magnetization(p),
                                        return_full_trajectory=False)[:, WATER_Z])


# 121 points -> step 0.1 ppm, so +DW and -DW both land EXACTLY on the grid and are exact
# mirror images (index i and n-1-i). With a 0.2 ppm step, argmin|ppm - 3.5| ties between 3.4
# and 3.6 and takes the first, while argmin|ppm + 3.5| takes -3.6 -- comparing 3.4 against
# -3.6 and manufacturing a 3.3e-3 "asymmetry" that has nothing to do with the physics.
PPM = np.linspace(-6.0, 6.0, 121)
I_POS = int(np.argmin(np.abs(PPM - DW)))
I_NEG = PPM.size - 1 - I_POS
assert PPM[I_POS] == pytest.approx(DW) and PPM[I_NEG] == pytest.approx(-DW)


def test_far_wing_is_unsaturated():
    """Far from water and from the pool, nothing is saturated, so Z returns to ~1."""
    z = z_spectrum(params(), PPM)
    assert 0.95 < z[0] < 1.01, f"Z(-6 ppm) = {z[0]:.4f}, expected close to 1"


def test_water_is_the_global_minimum():
    """Direct saturation of water is the deepest feature, and it sits AT water.

    The bound is +-0.5 ppm rather than exact equality, and that is physics rather than slop. The
    solver tracks ONE isochromat, and a 2 s CW block at 1 uT is several thousand degrees of
    flip, so on exact resonance the magnetisation nutates coherently instead of settling: the
    minimum lands on a neighbouring grid point (+0.1 ppm here, -0.1 ppm with no pool) depending
    on the phase the nutation happens to end at. A real voxel dephases through its intra-voxel
    B0 spread and does settle. This is the same effect that makes near-water offsets unusable
    for fitting a single-isochromat model.
    """
    z = z_spectrum(params(), PPM)
    assert abs(PPM[int(np.argmin(z))]) <= 0.5
    assert z.min() < z[I_POS], "the CEST dip is deeper than direct water saturation"


def test_no_pool_gives_a_symmetric_spectrum():
    """With f = 0 there is nothing off-resonance, so the spectrum is even in ppm.

    This is the test that catches a ppm/Hz scale error or a sign flip on the RF offset: those
    break the symmetry immediately, while a golden file would happily enshrine them.
    """
    z = z_spectrum(params(f=0.0), PPM)
    assert np.abs(z - z[::-1]).max() < 2e-3


def test_pool_creates_a_dip_on_its_own_side():
    """A pool at +3.5 ppm must reduce Z at +3.5 and leave -3.5 essentially alone."""
    z = z_spectrum(params(), PPM)
    assert z[I_POS] < z[I_NEG], f"Z(+{DW}) = {z[I_POS]:.4f} not below Z(-{DW}) = {z[I_NEG]:.4f}"
    assert z[I_NEG] - z[I_POS] > 0.05, "MTRasym at the pool offset is implausibly small"


def test_saturation_increases_with_b1_and_with_time():
    """Both QUESP and QUEST axes must move the spectrum in the expected direction."""
    p = params()
    assert z_spectrum(p, PPM, b1_uT=2.0).mean() < z_spectrum(p, PPM, b1_uT=1.0).mean()
    assert z_spectrum(p, PPM, tsat_s=4.0).mean() < z_spectrum(p, PPM, tsat_s=1.0).mean()


def test_zero_exchange_rate_removes_the_cest_effect():
    """k = 0 means no label transfer: the pool is present but invisible in water."""
    z = z_spectrum(params(k=0.0), PPM)
    assert abs(z[I_NEG] - z[I_POS]) < 2e-3, "a non-exchanging pool still produced an asymmetry"


@pytest.mark.parametrize("b0", [1.5, 2.8936, 7.0])
def test_ppm_axis_scales_with_field(b0):
    """The dip must sit at the same PPM at every field -- i.e. Hz/ppm is applied, not assumed.

    The pool is located against an f = 0 BASELINE rather than as the global minimum of Z. At
    1.5 T, 0.5 ppm is only 32 Hz from water, so direct saturation is still the deepest point in
    the window and `argmin(Z)` finds water, not the pool. Subtracting the pool-free spectrum
    removes that background; what is left peaks at the pool. It peaks slightly BELOW 3.5 ppm at
    low field (3.25 at 1.5 T, 3.38 at 2.9 T, 3.50 at 7 T) because the direct-saturation slope
    pulls it inward -- real physics, and the reason the tolerance is 0.4 rather than 0.05.
    """
    ppm = np.linspace(0.5, 6.0, 45)
    ev = cw_saturation(ppm, b1_uT=1.0, tsat_s=2.0, b0_T=b0)
    p_with = params()._replace(b0=np.float32(b0))
    p_none = params(f=0.0)._replace(b0=np.float32(b0))
    z_with, z_none = (np.asarray(simulate_spectrum(q, ev, get_initial_magnetization(q),
                                                   return_full_trajectory=False)[:, WATER_Z])
                      for q in (p_with, p_none))
    peak_ppm = ppm[int(np.argmax(z_none - z_with))]
    assert abs(peak_ppm - DW) < 0.4, f"pool found at {peak_ppm:.2f} ppm, expected ~{DW}"
    assert hz_per_ppm(b0) == pytest.approx(42.5764 * b0, rel=1e-6)
