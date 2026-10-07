"""fit_joint, end to end, against a truth the test itself defines (as test_fitting.py).

What only a joint fit can do is tested by building data that needs it: samples that share a water R2
(one value for all of them), and a second pool that only some samples contain (free there, absent
elsewhere). The model simulates the data and the fitter has to get the parameters back.

And what it must NOT change: with a plain spec -- every parameter per sample, free everywhere -- the
samples are independent, fit_joint runs fit's algorithm, and the two give the same numbers.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmc_jax.fitting import Experiment, FitSpec, FreeParam, fit, fit_joint, levenberg_marquardt
from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state, simulate_spectrum
from bmc_jax.sequences import cw_saturation

PPM = np.linspace(-6.0, 6.0, 41)
LADDER = (0.5, 1.0, 1.5, 2.0)          # uT


def _ladder(p):
    return jax.tree.map(lambda *a: jnp.concatenate(a, axis=0),
                        *[cw_saturation(PPM, b1_uT=b, tsat_s=2.0, b0_T=p.system.b0) for b in LADDER])


def _batch(jp, n, **fields):
    """n samples of `jp`, with some fields set per sample."""
    base = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (n,) + jnp.asarray(a).shape), jp)
    return base._replace(**{k: jnp.asarray(v, base.water_r1.dtype) for k, v in fields.items()})


def _data(base, events, mz):
    one = lambda i: jax.tree.map(lambda a: a[i], base)
    n = base.water_r1.shape[0]
    m0 = jnp.stack([initial_state(one(i)) for i in range(n)])
    z = jnp.stack([simulate_spectrum(one(i), events, m0[i], return_full_trajectory=False)[:, mz]
                   for i in range(n)])
    return m0, z


K = FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=2000.0, x0=150.0)
F = FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-5, hi=0.01, x0=0.0005)
R2W = FreeParam("r2w", "water_r2", None, scale=10.0, lo=1.0, hi=100.0, x0=14.0, scope="shared")
F_NOE = FreeParam("f_noe", "cest_f", 1, scale=0.001, lo=0.0, hi=0.03, x0=0.002, samples=(0, 1))


@pytest.fixture(scope="module")
def amide():
    """Three amide samples, k 40 / 120 / 300 /s, the config's f and water R2 (20 /s)."""
    p = example_config("amide")
    jp = to_sim_params(p)
    events = _ladder(p)
    true = _batch(jp, 3, cest_k=[[40.0], [120.0], [300.0]])
    m0, z = _data(true, events, p.mz_loc)
    return p, jp, true, events, m0, z


@pytest.fixture(scope="module")
def noe():
    """amide + NOE: the NOE pool in samples 0 and 1 only (f 0.0045 / 0.009); sample 2 has none (f 0)."""
    p = example_config("amide_noe")
    jp = to_sim_params(p)
    events = _ladder(p)
    true = _batch(jp, 3, cest_k=[[50.0, 20.0], [150.0, 20.0], [300.0, 20.0]],
                  cest_f=[[0.0009, 0.0045], [0.0009, 0.009], [0.0009, 0.0]])
    m0, z = _data(true, events, p.mz_loc)
    return p, jp, true, events, m0, z


def test_a_plain_spec_gives_fits_numbers(amide):
    """Every parameter per sample and free everywhere: independent samples, fit's algorithm, fit's result."""
    p, jp, true, events, m0, z = amide
    exp = Experiment(true, events, m0, p.mz_loc)
    spec = FitSpec([K, F])
    a, b = fit(exp, spec, z, n_iter=20), fit_joint(exp, spec, z, n_iter=20)
    np.testing.assert_allclose(b.x, a.x, rtol=1e-6, atol=0)
    np.testing.assert_allclose(b.z_fit, a.z_fit, rtol=1e-6, atol=1e-7)
    assert b.free.all() and np.isfinite(b.sd).all()


def test_a_shared_value_is_recovered(amide):
    """Water R2 shared by three samples, starting 30 % off: one value, the truth, for all of them."""
    p, jp, true, events, m0, z = amide
    start = true._replace(water_r2=jnp.full(3, 14.0, true.water_r1.dtype))   # what the params carry
    res = fit_joint(Experiment(start, events, m0, p.mz_loc), FitSpec([K, F, R2W]), z, n_iter=30)
    assert np.all(res["r2w"] == res["r2w"][0]), "a shared value is ONE value"
    assert res["r2w"][0] == pytest.approx(20.0, rel=5e-3)
    np.testing.assert_allclose(res["k"], [40.0, 120.0, 300.0], rtol=3e-2)
    assert np.all(res.sd[:, 2] == res.sd[0, 2]) and np.isfinite(res.sd).all()
    assert res.cost[-1] < res.cost[0] and res.nan_steps == 0


def test_a_shared_value_survives_a_chunked_jacobian(amide):
    """One sample per device call (the NaN-Jacobian workaround) still finds the same truth."""
    p, jp, true, events, m0, z = amide
    start = true._replace(water_r2=jnp.full(3, 14.0, true.water_r1.dtype))
    res = fit_joint(Experiment(start, events, m0, p.mz_loc), FitSpec([K, F, R2W]), z, n_iter=30, max_chunk=1)
    assert res["r2w"][0] == pytest.approx(20.0, rel=5e-3)


@pytest.mark.parametrize("shared", [False, True], ids=["independent", "coupled"])
def test_a_pool_free_in_some_samples_only(noe, shared):
    """The NOE pool fitted where it exists; where it does not, its f stays exactly what the params carry."""
    p, jp, true, events, m0, z = noe
    start = true._replace(water_r2=jnp.full(3, 14.0, true.water_r1.dtype)) if shared else true
    spec = FitSpec([K, F, F_NOE] + ([R2W] if shared else []))
    res = fit_joint(Experiment(start, events, m0, p.mz_loc), spec, z, n_iter=30)
    np.testing.assert_allclose(res["f_noe"][:2], [0.0045, 0.009], rtol=1e-2)
    assert res["f_noe"][2] == 0.0, "not free in sample 2: the carried value, exactly"
    assert not res.free[2, 2] and np.isnan(res.sd[2, 2]) and np.isfinite(res.sd[:2, 2]).all()
    assert not res.at_bounds[2, 2], "a carried 0 on the lower bound is not a railed fit"
    if shared:
        assert res["r2w"][0] == pytest.approx(20.0, rel=5e-3)


def test_fit_refuses_what_it_cannot_do(amide):
    p, jp, true, events, m0, z = amide
    with pytest.raises(ValueError, match="fit_joint"):
        fit(Experiment(true, events, m0, p.mz_loc), FitSpec([K, R2W]), z)


def test_samples_must_exist(amide):
    p, jp, true, events, m0, z = amide
    bad = FreeParam("f2", "cest_f", 0, 0.001, 0.0, 0.03, 0.002, samples=(0, 7))
    with pytest.raises(ValueError, match="do not exist"):
        fit_joint(Experiment(true, events, m0, p.mz_loc), FitSpec([K, bad]), z, n_iter=1)


def test_levenberg_marquardt_holds_what_is_not_free():
    """No physics: a held entry stays at its start, the rest solve as before."""
    resid = lambda x: jnp.array([x[0] - 1.0, x[1] - 2.0, 0.1 * x[0] * x[1]])
    x0, lo, hi = jnp.array([0.0, 5.0]), jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0])
    held = levenberg_marquardt(resid, x0, lo, hi, n_iter=30, free=jnp.array([1.0, 0.0]))
    assert float(held[1]) == 5.0
    both = levenberg_marquardt(resid, x0, lo, hi, n_iter=30)
    ones = levenberg_marquardt(resid, x0, lo, hi, n_iter=30, free=jnp.array([1.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(both), np.asarray(ones))
