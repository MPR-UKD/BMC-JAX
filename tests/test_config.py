"""Reading a Pulseq-CEST config. Self-contained -- uses only the configs shipped in the package.

These tests deliberately do NOT sweep a large corpus. The config format has a small number of
branches -- rates or times, MT present or absent, aliases, defaults -- and four configs cover
them. A thousand files would exercise the same branches a thousand times and tell you no more.

The one test worth reading twice is `test_mz_loc_matches_the_simulator`: it checks the index of
water Mz against where the solver actually puts it, rather than against the formula that computed
it. An index asserted against its own formula is not a test.
"""
import numpy as np
import pytest
import yaml

from bmc_jax.io import (CESTPool, MTPool, Options, Parameters, System, WaterPool,
                        example_config, example_config_names, from_yaml)

ALL = example_config_names()


# --- the shipped configs ---------------------------------------------------------------------

def test_every_shipped_config_loads():
    assert set(ALL) == {"amide", "amide_noe", "amide_noe_mt", "phantom_creatine"}
    for name in ALL:
        p = example_config(name)
        assert p.water_pool.f == 1.0
        assert p.system.b0 > 0
        assert all(c.k >= 0 for c in p.cest_pools)


def test_the_branches_are_all_covered():
    """Rates and times, MT and no MT, one pool and several -- across the four configs."""
    assert example_config("amide").num_cest_pools == 1
    assert example_config("amide_noe").num_cest_pools == 2
    assert example_config("amide_noe_mt").mt_pool.lineshape == "superlorentzian"
    assert example_config("amide_noe").mt_pool is None
    # phantom_creatine gives r1/r2 directly instead of t1/t2
    assert example_config("phantom_creatine").water_pool.r1 == pytest.approx(0.3039)


def test_times_and_rates_are_the_same_thing():
    p = example_config("amide")
    assert p.water_pool.r1 == pytest.approx(1 / 1.3)
    assert p.water_pool.T1 == pytest.approx(1.3)
    assert p.cest_pools[0].dw == pytest.approx(3.5)


# --- what must be rejected -------------------------------------------------------------------

def test_both_a_rate_and_a_time_is_an_error():
    """Not a silent preference: if r1 and t1 disagree, one of them is a mistake."""
    with pytest.raises(ValueError, match="not both"):
        WaterPool(f=1.0, r1=1.0, t1=1.0, r2=20.0)


def test_neither_a_rate_nor_a_time_is_an_error():
    with pytest.raises(ValueError, match="not neither"):
        WaterPool(f=1.0, r2=20.0)


@pytest.mark.parametrize("bad", [{"f": 1.5}, {"f": -0.1}])
def test_pool_fraction_must_be_a_fraction(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        WaterPool(t1=1.0, t2=0.05, **bad)


def test_lineshape_must_be_one_we_implement():
    with pytest.raises(ValueError, match="lineshape"):
        MTPool(f=0.05, t1=1.0, t2=1e-5, k=30, dw=-2.5, lineshape="gaussian")


def test_negative_rates_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        WaterPool(f=1.0, r1=-1.0, r2=20.0)


def test_a_config_without_water_is_rejected():
    with pytest.raises(ValueError, match="water_pool"):
        Parameters.from_dict({"b0": 3})


# --- defaults and aliases --------------------------------------------------------------------

def test_scanner_defaults_are_applied():
    """bmctool requires all four; we default the three that have an obvious nominal value."""
    p = Parameters.from_dict({"water_pool": {"f": 1.0, "t1": 1.3, "t2": 0.05}, "b0": 3})
    assert p.system.gamma == pytest.approx(267.5153)
    assert p.system.b0_inhom == 0.0
    assert p.system.rel_b1 == 1.0
    assert p.options.max_pulse_samples == 500


def test_alternative_spellings_are_accepted():
    p = Parameters.from_dict({"water_pool": {"f": 1.0, "t1": 1.3, "t2": 0.05},
                              "b0": 3, "relb1": 0.9, "b0_inhomogeneity": 0.2, "max_samples": 100})
    assert (p.system.rel_b1, p.system.b0_inhom, p.options.max_pulse_samples) == (0.9, 0.2, 100)


# --- it has to actually drive the solver -----------------------------------------------------

def test_config_to_spectrum_end_to_end():
    """The point of the whole file: .yaml -> Z-spectrum with no bmctool anywhere."""
    from bmc_jax.jax_solver import (convert_params, cw_saturation, get_initial_magnetization,
                                    simulate_spectrum)
    p = example_config("amide")
    jp = convert_params(p)
    ppm = np.linspace(-6, 6, 61)
    ev = cw_saturation(ppm, b1_uT=1.0, tsat_s=2.0, b0_T=p.system.b0)
    z = np.asarray(simulate_spectrum(jp, ev, get_initial_magnetization(jp),
                                     return_full_trajectory=False)[:, p.mz_loc])
    assert z.shape == ppm.shape
    assert 0.0 < z.min() < z.max() < 1.05
    i = int(np.argmin(np.abs(ppm - 3.5)))
    assert z[i] < z[ppm.size - 1 - i], "the amide pool produced no asymmetry"


@pytest.mark.parametrize("name", ALL)
def test_mz_loc_matches_the_simulator(name):
    """Check the water-Mz index against WHERE THE SOLVER PUTS IT, not against its own formula.

    mz_loc = 2 + 2*n_cest is an index into the solver's state layout. Asserting it equals
    2 + 2*n_cest tests arithmetic. Simulating far off resonance, where nothing is saturated and
    water Mz must be the equilibrium value, tests the layout.
    """
    from bmc_jax.jax_solver import (convert_params, cw_saturation, get_initial_magnetization,
                                    simulate_spectrum)
    p = example_config(name)
    jp = convert_params(p)
    ev = cw_saturation([-300.0], b1_uT=0.0, tsat_s=0.1, b0_T=p.system.b0)
    m = np.asarray(simulate_spectrum(jp, ev, get_initial_magnetization(jp),
                                     return_full_trajectory=False))[0]
    assert m[p.mz_loc] == pytest.approx(p.water_pool.f, abs=1e-3), (
        f"index {p.mz_loc} holds {m[p.mz_loc]:.4f}, not water Mz ({p.water_pool.f})")
