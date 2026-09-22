"""Fitting, end to end, against a truth the test itself defines. No files, no reference copy.

The model simulates the data and then the fitter has to get the parameters back. That is a real
test of the whole chain -- spec, residual, optimiser, batching -- and it needs nothing external,
unlike a comparison against a stored result or another implementation.

WHAT THE TOLERANCES MEAN
  k is recovered to a few percent, f to a fraction of a percent, and the two are NOT equally
  well determined. Saturation efficiency is alpha = w1^2 / (w1^2 + k(k + R2s)): a ladder only
  separates k from f where alpha actually moves across it. The slowest pool here (k = 30 /s) is
  already saturated at the bottom rung, so its k is the loosest number in the fit. That is
  physics, not solver error, and the tolerances say so rather than hiding it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmc_jax.fitting import FitSpec, Slot, fit_phantom_lm, lm_minimize
from bmc_jax.io import example_config
from bmc_jax.jax_solver import (convert_params, cw_saturation, get_initial_magnetization,
                                simulate_spectrum)

PPM = np.linspace(-6.0, 6.0, 41)
LADDER = (0.5, 1.0, 1.5, 2.0)          # uT -- a QUESP ladder
K_TRUE = np.linspace(30.0, 400.0, 8)


def _experiment():
    """Eight samples differing only in exchange rate, simulated over the whole ladder.

    The per-level forward models are concatenated along the OFFSET axis and the measured Z the
    same way, so one residual spans the ladder. A single saturation power constrains the exchange
    FLUX k*f, not its two factors -- fitting k and f from one level is not slow, it is impossible.
    """
    p = example_config("amide")
    jp = convert_params(p)
    events = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0),
                          *[cw_saturation(PPM, b1_uT=b, tsat_s=2.0, b0_T=p.system.b0)
                            for b in LADDER])
    n = K_TRUE.size
    base = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (n,) + jnp.asarray(a).shape), jp)
    base = base._replace(cest_k=jnp.asarray(K_TRUE[:, None], jnp.float32))
    M0 = jnp.stack([get_initial_magnetization(jax.tree.map(lambda a, i=i: a[i], base))
                    for i in range(n)])
    Z = jnp.stack([simulate_spectrum(jax.tree.map(lambda a, i=i: a[i], base), events, M0[i],
                                     return_full_trajectory=False)[:, p.mz_loc] for i in range(n)])
    return p, base, events, M0, Z


SPEC = FitSpec([Slot("k", "cest_k", 0, scale=100.0, lo=5.0, hi=2000.0, x0=150.0),
                Slot("f", "cest_f", 0, scale=0.001, lo=1e-5, hi=0.01, x0=0.0005)])


@pytest.fixture(scope="module")
def fitted():
    p, base, events, M0, Z = _experiment()
    X = np.asarray(fit_phantom_lm(base, events, M0, p.mz_loc, SPEC, Z,
                                  n_iter=30, skip_first=False, base_batched=True))
    return p, X


def test_exchange_rate_is_recovered(fitted):
    _, X = fitted
    rel = np.abs(X[:, 0] / K_TRUE - 1.0)
    assert rel.max() < 0.05, f"worst k error {rel.max():.3f} at k = {K_TRUE[rel.argmax()]}"
    assert rel[K_TRUE > 100].max() < 0.005, "the well-bracketed pools should be far better"


def test_pool_fraction_is_recovered(fitted):
    p, X = fitted
    f_true = p.cest_pools[0].f
    assert np.abs(X[:, 1] / f_true - 1.0).max() < 0.03


def test_the_flux_is_better_determined_than_its_factors(fitted):
    """k*f is what one saturation power actually constrains; it should be the tighter number."""
    p, X = fitted
    flux_err = np.abs((X[:, 0] * X[:, 1]) / (K_TRUE * p.cest_pools[0].f) - 1.0).max()
    k_err = np.abs(X[:, 0] / K_TRUE - 1.0).max()
    assert flux_err <= k_err + 1e-9


def test_bounds_are_respected(fitted):
    _, X = fitted
    assert (X[:, 0] >= SPEC.lo_phys[0]).all() and (X[:, 0] <= SPEC.hi_phys[0]).all()
    assert (X[:, 1] >= SPEC.lo_phys[1]).all() and (X[:, 1] <= SPEC.hi_phys[1]).all()


def test_lm_minimize_works_without_any_physics():
    """The optimiser is a general bounded least-squares routine. Prove it on Rosenbrock.

    If this needs anything from the solver, optimize.py is not really a separate layer.
    """
    def resid(x):
        return jnp.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    x = np.asarray(lm_minimize(resid, jnp.array([-1.2, 1.0]), jnp.array([-5.0, -5.0]),
                               jnp.array([5.0, 5.0]), n_iter=200))
    assert np.allclose(x, [1.0, 1.0], atol=1e-3), f"converged to {x}, expected (1, 1)"
