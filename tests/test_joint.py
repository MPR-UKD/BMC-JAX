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


# ---- factors, offsets, the start state, one value per experiment ---------------------------------------

M_SHARED = FreeParam("m0", "m_start", None, scale=1.0, lo=0.1, hi=1.0, x0=0.9, scope="shared")
R2_FACTOR = FreeParam("r2f", "water_r2", None, scale=1.0, lo=0.5, hi=1.5, x0=1.0, scope="shared", mode="factor")
DB0 = FreeParam("db0", "b0_inhom", None, scale=0.01, lo=-0.1, hi=0.1, x0=0.0, scope="block", mode="add")


def test_a_shared_factor_and_start_state_are_recovered(amide):
    """Measured R2 per sample 25 % too high (truth = 0.8 x carried) and the sequence starting at 0.6 of
    equilibrium: one factor and one start state, both found, with the carried R2 left per sample."""
    p, jp, true, events, m0, z = amide
    r2_true = jnp.asarray([16.0, 20.0, 24.0], true.water_r1.dtype)
    true2 = true._replace(water_r2=r2_true)
    one = lambda b, i: jax.tree.map(lambda a: a[i], b)
    m06 = jnp.stack([initial_state(one(true2, i)) * 0.6 for i in range(3)])
    z2 = jnp.stack([simulate_spectrum(one(true2, i), events, m06[i], return_full_trajectory=False)[:, p.mz_loc]
                    for i in range(3)])
    carried = true2._replace(water_r2=r2_true / 0.8)                       # what was "measured"
    m_eq = jnp.stack([initial_state(one(carried, i)) for i in range(3)])   # the experiment assumes equilibrium
    res = fit_joint(Experiment(carried, events, m_eq, p.mz_loc), FitSpec([K, F, R2_FACTOR, M_SHARED]), z2,
                    n_iter=40)
    assert res["r2f"][0] == pytest.approx(0.8, rel=5e-3)
    assert res["m0"][0] == pytest.approx(0.6, rel=5e-3)
    np.testing.assert_allclose(res["k"], [40.0, 120.0, 300.0], rtol=3e-2)


def test_a_start_state_free_in_some_samples_only(amide):
    """M_start 0.7 / 0.5 in samples 0 and 1 (fitted), sample 2 at the start state the experiment gives."""
    p, jp, true, events, m0, z = amide
    one = lambda b, i: jax.tree.map(lambda a: a[i], b)
    ms = [0.7, 0.5, 1.0]
    m_true = jnp.stack([initial_state(one(true, i)) * ms[i] for i in range(3)])
    z2 = jnp.stack([simulate_spectrum(one(true, i), events, m_true[i], return_full_trajectory=False)[:, p.mz_loc]
                    for i in range(3)])
    m_each = FreeParam("m0", "m_start", None, scale=1.0, lo=0.1, hi=1.0, x0=0.9, samples=(0, 1))
    res = fit_joint(Experiment(true, events, m0, p.mz_loc), FitSpec([K, F, m_each]), z2, n_iter=40)
    np.testing.assert_allclose(res["m0"][:2], [0.7, 0.5], rtol=5e-3)
    assert res["m0"][2] == pytest.approx(1.0) and not res.free[2, 2]
    np.testing.assert_allclose(res["k"], [40.0, 120.0, 300.0], rtol=3e-2)


def test_one_offset_per_experiment_is_recovered(amide):
    """A B0 drift between the rungs (0, +0.03, -0.02, +0.01 ppm) on top of each sample's own B0."""
    p, jp, true, events, m0, z = amide
    own = jnp.asarray([0.0, 0.01, -0.01], true.water_r1.dtype)
    drift = [0.0, 0.03, -0.02, 0.01]
    n_per = PPM.size
    one = lambda b, i: jax.tree.map(lambda a: a[i], b)
    blocks = []
    for b, d in enumerate(drift):
        ev = jax.tree.map(lambda a: a[b * n_per:(b + 1) * n_per], events)
        blocks.append(jnp.stack([simulate_spectrum(one(true, i)._replace(b0_inhom=own[i] + d), ev, m0[i],
                                                   return_full_trajectory=False)[:, p.mz_loc] for i in range(3)]))
    z2 = jnp.concatenate(blocks, axis=1)
    res = fit_joint(Experiment(true._replace(b0_inhom=own), events, m0, p.mz_loc), FitSpec([K, F, DB0]), z2,
                    n_iter=40, n_per_block=n_per)
    np.testing.assert_allclose(res["db0"], drift, atol=2e-3)
    assert np.isnan(res.x[:, 2]).all() and res.blocks["db0"]["sd"].shape == (4,)
    np.testing.assert_allclose(res["k"], [40.0, 120.0, 300.0], rtol=3e-2)


@pytest.mark.parametrize("extra", [R2_FACTOR, M_SHARED, DB0], ids=["factor", "m_start", "block"])
def test_fit_refuses_factors_offsets_start_state_and_blocks(amide, extra):
    p, jp, true, events, m0, z = amide
    with pytest.raises(ValueError, match="fit_joint"):
        fit(Experiment(true, events, m0, p.mz_loc), FitSpec([K, extra]), z)


# ---- the B0 spread inside a sample ----------------------------------------------------------------------

from bmc_jax.simulation import B0Spread, simulate_spectrum_spread  # noqa: E402

SIGMA = FreeParam("sig", "b0_spread", None, scale=10.0, lo=0.0, hi=60.0, x0=10.0, scope="shared")
B1F = FreeParam("b1f", "rel_b1", None, scale=1.0, lo=0.7, hi=1.3, x0=1.0, scope="block", mode="factor")
_one = lambda b, i: jax.tree.map(lambda a: a[i], b)


@pytest.mark.parametrize("shape", ["gauss", "lorentz"])
def test_a_spread_of_zero_width_is_one_isochromat(amide, shape):
    p, jp, true, events, m0, z = amide
    a = simulate_spectrum(_one(true, 0), events, m0[0], return_full_trajectory=False)
    b = simulate_spectrum_spread(_one(true, 0), events, m0[0], 0.0, B0Spread(shape=shape, nodes=5))
    np.testing.assert_allclose(b, a, rtol=0, atol=2e-6)


def test_spread_nodes_and_weights():
    """Gauss: weights sum to 1, SD 1. Lorentz cut at +-c: E[1 / (1 + u^2)] = 1/2 + sin(2T) / 4T, T = atan(c)."""
    u, w = B0Spread(shape="gauss", nodes=5).nodes_weights()
    assert w.sum() == pytest.approx(1.0) and np.sum(w * u ** 2) == pytest.approx(1.0)
    u, w = B0Spread(shape="lorentz", nodes=9, cut=20.0).nodes_weights()
    T = np.arctan(20.0)
    assert np.sum(w / (1 + u ** 2)) == pytest.approx(0.5 + np.sin(2 * T) / (4 * T), rel=1e-10)
    assert np.all(w > 0) and np.allclose(u, -u[::-1])
    with pytest.raises(ValueError, match="shape"):
        B0Spread(shape="box")


def test_a_node_is_the_sample_shifted_by_sigma(amide):
    """Two Gauss-Hermite nodes sit at +-1 sigma with weight 1/2: the mean of the sample shifted by +-sigma (Hz,
    converted with the sample's own field: 42.58 Hz per ppm per tesla)."""
    p, jp, true, events, m0, z = amide
    one = _one(true, 0)
    hz = float(one.gamma * one.b0 / (2 * np.pi))
    assert hz == pytest.approx(42.577 * float(one.b0), rel=1e-3)
    s = 12.0
    sh = lambda d: simulate_spectrum(one._replace(b0_inhom=one.b0_inhom + d), events, m0[0],
                                     return_full_trajectory=False)
    two = simulate_spectrum_spread(one, events, m0[0], s, B0Spread(nodes=2))
    np.testing.assert_allclose(two, 0.5 * (sh(s / hz) + sh(-s / hz)), rtol=0, atol=2e-6)


def test_a_held_spread_per_sample(amide):
    """No b0_spread slot: every sample simulated with its own given width, nothing fitted."""
    p, jp, true, events, m0, z = amide
    sp = B0Spread(sigma_hz=(0.0, 10.0, 30.0))
    x0 = np.array([[40.0, 0.0009009], [120.0, 0.0009009], [300.0, 0.0009009]])     # the truth: model = data's params
    d = fit_joint(Experiment(true, events, m0, p.mz_loc), FitSpec([K, F]), z, x0_phys=x0, spread=sp,
                  _internals=True)
    zm = np.asarray(d["resid"](d["x0"])).reshape(z.shape) + np.asarray(z)
    for i, s in enumerate(sp.sigma_hz):           # 1e-5: k and f pass through x / scale * scale in float32
        want = simulate_spectrum_spread(_one(true, i), events, m0[i], s, sp)[:, p.mz_loc]
        np.testing.assert_allclose(zm[i], want, rtol=0, atol=1e-5)


def test_a_shared_spread_is_recovered(amide):
    """Data with a 25 Hz Gaussian spread in every sample; one width fitted from 10 Hz, with k and f."""
    p, jp, true, events, m0, z = amide
    sp = B0Spread(nodes=5)
    z2 = jnp.stack([simulate_spectrum_spread(_one(true, i), events, m0[i], 25.0, sp)[:, p.mz_loc] for i in range(3)])
    res = fit_joint(Experiment(true, events, m0, p.mz_loc), FitSpec([K, F, SIGMA]), z2, n_iter=30, spread=sp)
    assert res["sig"][0] == pytest.approx(25.0, rel=2e-2)
    np.testing.assert_allclose(res["k"], [40.0, 120.0, 300.0], rtol=3e-2)


def test_b0_spread_needs_fit_joint_and_a_spread(amide):
    p, jp, true, events, m0, z = amide
    exp = Experiment(true, events, m0, p.mz_loc)
    with pytest.raises(ValueError, match="fit_joint"):
        fit(exp, FitSpec([K, SIGMA]), z)
    with pytest.raises(ValueError, match="spread"):
        fit_joint(exp, FitSpec([K, SIGMA]), z, n_iter=1)
    with pytest.raises(ValueError, match="b0_spread"):
        FreeParam("s", "b0_spread", None, 1.0, 0.0, 1.0, 0.5, mode="factor")


# ---- the assembled Jacobian is the derivative, for every kind of value ----------------------------------

M_SOME = FreeParam("m0", "m_start", None, scale=1.0, lo=0.1, hi=1.0, x0=0.9, samples=(0, 1))
JAC_CASES = {                       # slots, spread, blocks ("full" / "split" / None), chunk
    "plain": ([K, F], None, None, None),
    "shared": ([K, F, R2W], None, None, None),
    "factor + start state": ([K, F, R2_FACTOR, M_SHARED], None, None, None),
    "some samples": ([K, F, M_SOME], None, None, None),
    "block offset": ([K, F, DB0], None, "full", None),
    "block offset, split": ([K, F, DB0], None, "split", None),
    "block factor + shared, split": ([K, F, B1F, R2W], None, "split", None),
    "spread": ([K, F, SIGMA], B0Spread(nodes=3), None, None),
    "block + spread, split, chunked": ([K, F, B1F, SIGMA], B0Spread(nodes=3, shape="lorentz"), "split", 1),
}


@pytest.mark.parametrize("case", list(JAC_CASES))
def test_the_joint_jacobian_is_the_derivative(amide, case):
    """fit_joint builds its Jacobian per sample and scatters it ("split": a per-block slot as ONE tangent split by
    rows). It must equal jax.jacfwd of the whole residual, column by column, away from the start.
    Full matmul precision: by default a GPU multiplies float32 matrices in TF32, and the two Jacobians, built in a
    different order, then differ by ~2e-4 (RTX 3070) for no fault of the assembly. On CPU this changes nothing."""
    p, jp, true, events, m0, z = amide
    slots, sp, blocks, chunk = JAC_CASES[case]
    with jax.default_matmul_precision("highest"):
        d = fit_joint(Experiment(true, events, m0, p.mz_loc), FitSpec(slots), z, spread=sp, max_chunk=chunk,
                      n_per_block=PPM.size if blocks else None, block_jacobian=blocks or "full", _internals=True)
        x = d["x0"] + 0.02 * jnp.linspace(-1.0, 1.0, d["x0"].size, dtype=d["x0"].dtype)
        J, Ja = np.asarray(d["jac"](x)), np.asarray(jax.jacfwd(d["resid"])(x))
    rel = np.linalg.norm(J - Ja, axis=0) / np.maximum(np.linalg.norm(Ja, axis=0), 1e-30)
    assert rel.max() < 1e-4, f"{case}: columns {np.flatnonzero(rel >= 1e-4)} off by {rel.max():.1e}"
