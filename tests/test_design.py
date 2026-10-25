"""bmc_jax.design: the Cramer-Rao bound and protocol optimisation, checked against what must hold.

The bound is F^-1 with F = J^T J / sigma^2, so some of its properties follow from the formula alone:
it scales with the noise, it falls by sqrt(2) when the same data are measured twice, and a parameter
estimated alone is bounded by 1/sqrt(F_jj), never above the bound with the other parameters free.
The optimiser must lower the cost and keep the design within its bounds.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmc_jax.design import crlb, optimize_design
from bmc_jax.fitting import Experiment, FitSpec, FreeParam
from bmc_jax.io import example_config
from bmc_jax.sequences import cw_saturation
from bmc_jax.simulation import initial_state, tile_params, to_sim_params

PPM = np.linspace(-6.0, 6.0, 31)
SIGMA = 0.002
K = FreeParam("k", "cest_k", 0, scale=100.0, lo=5.0, hi=5000.0, x0=150.0)
F = FreeParam("f", "cest_f", 0, scale=0.001, lo=1e-6, hi=0.05, x0=0.0009)


@pytest.fixture(scope="module")
def amide():
    """The amide config and three rungs of 2 s continuous-wave saturation, each built at 1 uT."""
    p = example_config("amide")
    jp = to_sim_params(p)
    rungs = [cw_saturation(PPM, b1_uT=1.0, tsat_s=2.0, b0_T=p.system.b0) for _ in range(3)]
    return p, jp, initial_state(jp), rungs


def _ladder(amide, b1):
    """The three rungs at the amplitudes b1 (uT): the amplitude scales rf_amp linearly."""
    p, jp, m0, rungs = amide
    ev = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0),
                      *[e._replace(rf_amp=e.rf_amp * b) for e, b in zip(rungs, b1)])
    return Experiment(jp, ev, m0, p.mz_loc)


def test_the_bound_scales_with_the_noise(amide):
    exp = _ladder(amide, (0.5, 1.0, 2.0))
    a, b = crlb(exp, FitSpec([K, F]), SIGMA), crlb(exp, FitSpec([K, F]), 2 * SIGMA)
    np.testing.assert_allclose(b.sd, 2 * a.sd, rtol=1e-5)
    np.testing.assert_allclose(b.corr, a.corr, atol=1e-6)
    assert a["k"] == a.sd[0] and a.rel_sd[0] == pytest.approx(a.sd[0] / 150.0)


def test_twice_the_data_lowers_the_bound_by_sqrt2(amide):
    once = _ladder(amide, (0.5, 1.0, 2.0))
    twice = Experiment.stack([once, once])
    a, b = crlb(once, FitSpec([K, F]), SIGMA), crlb(twice, FitSpec([K, F]), SIGMA)
    np.testing.assert_allclose(b.sd, a.sd / np.sqrt(2.0), rtol=1e-4)


def test_a_parameter_alone_is_bounded_by_its_own_fisher_entry(amide):
    exp = _ladder(amide, (0.5, 1.0, 2.0))
    both, alone = crlb(exp, FitSpec([K, F]), SIGMA), crlb(exp, FitSpec([K]), SIGMA)
    # 5e-3: in float32 the two Jacobians come from different compiled kernels, and the 2 s saturation steps
    # carry about 1e-4 of rounding (tutorial 04); measured 9e-4 here
    assert alone.sd[0] == pytest.approx(1.0 / np.sqrt(both.fisher[0, 0]), rel=5e-3)
    assert alone.sd[0] < both.sd[0], "with f free as well, k cannot be known better"


def test_the_bound_needs_one_sample_and_plain_values(amide):
    exp = _ladder(amide, (0.5, 1.0, 2.0))
    with pytest.raises(ValueError, match="one sample"):
        crlb(Experiment(tile_params(exp.params, 2), exp.events, exp.m0, exp.mz_loc), FitSpec([K, F]), SIGMA)
    shared = FreeParam("r2w", "water_r2", None, scale=1.0, lo=0.5, hi=2.0, x0=1.0, scope="shared", mode="factor")
    with pytest.raises(ValueError, match="plain values"):
        crlb(exp, FitSpec([K, F, shared]), SIGMA)


def test_optimising_the_amplitudes_lowers_the_bound(amide):
    res = optimize_design(lambda d: _ladder(amide, d), FitSpec([K, F]), d0=[0.5, 1.0, 2.0], lo=[0.1] * 3,
                          hi=[3.0] * 3, sigma=SIGMA, truths=[[150.0, 0.0009]], n_steps=15, lr=0.1)
    assert res.cost[-1] < res.cost[0]
    assert np.all(res.design > 0.1) and np.all(res.design < 3.0)
    assert np.all(res.rel_sd_end <= res.rel_sd_start * 1.0001)
    with pytest.raises(ValueError, match="lo < d0 < hi"):
        optimize_design(lambda d: _ladder(amide, d), FitSpec([K, F]), d0=[0.05, 1.0, 2.0], lo=[0.1] * 3,
                        hi=[3.0] * 3, sigma=SIGMA, truths=[[150.0, 0.0009]], n_steps=1)
