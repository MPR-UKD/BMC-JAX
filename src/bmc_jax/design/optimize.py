"""optimize.py -- a protocol with a lower Cramer-Rao bound, found by gradient descent.

A design is a short vector d of protocol settings, for example one saturation amplitude per rung of a
QUESP ladder. The caller's function `build(d)` turns it into an Experiment, using JAX operations only,
so that the whole chain design -> events -> simulation -> Jacobian -> Fisher matrix -> bound can be
differentiated. The cost is the mean, over a set of assumed true values, of the squared relative
bound of the target parameters:

    cost(d) = mean over truths of  sum over targets of  (SD_j / x_j)^2.

Averaging over several true values makes the protocol good for a range of samples rather than for a
single guess. The gradient of the cost with respect to d is a derivative of a derivative (the bound
is built from the Jacobian), which JAX computes like any other.

The design stays within its bounds because it is written as d = lo + (hi - lo) * sigmoid(u), and u
is optimised with Adam. Everything the bound does not see is the caller's responsibility: keep the
scan time fixed in `build` (the same number of rungs, offsets and saturation time), express hardware
limits such as a maximum amplitude through `hi`, and remember that a lower bound helps only where the
noise, not the model, limits the fit.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .crlb import crlb, fisher_information


@dataclass(frozen=True)
class DesignResult:
    """The optimised design and how much it lowered the bound."""
    design: np.ndarray           # the optimised settings
    start: np.ndarray            # the settings it started from
    cost: np.ndarray             # the cost before every step, and after the last one
    targets: tuple               # names of the target parameters
    truths: np.ndarray           # the true values the cost averaged over, (n_truths, n_parameters)
    rel_sd_start: np.ndarray     # relative bound of the targets per truth, start design (n_truths, n_targets)
    rel_sd_end: np.ndarray       # the same for the optimised design


def optimize_design(build: Callable, spec, d0, lo, hi, sigma: float, truths,
                    targets: Optional[Sequence[str]] = None, n_steps: int = 200, lr: float = 0.05,
                    callback: Optional[Callable] = None) -> DesignResult:
    """Lower the Cramer-Rao bound of the target parameters by changing the design.

    Parameters
    ----------
    build
        `build(d) -> Experiment` for one sample, written with JAX operations so that it can be
        differentiated with respect to `d`.
    spec
        The parameters that will be estimated (plain values per sample), as for `fit` and `crlb`.
    d0, lo, hi
        The start design and its lower and upper bounds (same length). `d0` must lie strictly inside.
    sigma
        Standard deviation of the noise on every point (units of Z).
    truths
        True values to average the cost over, physical units, shape (n_truths, n_parameters).
    targets
        Names of the parameters whose bound is minimised. Default: every parameter of `spec`.
    n_steps, lr
        Adam steps and step size (in the unbounded variable u).
    callback
        Called as `callback(step, cost, design)` after every step.

    Returns
    -------
    DesignResult
    """
    names = [s.name for s in spec.slots]
    targets = tuple(names if targets is None else targets)
    idx = jnp.asarray([names.index(t) for t in targets])
    truths = np.atleast_2d(np.asarray(truths, float))
    d0, lo, hi = (np.asarray(a, float) for a in (d0, lo, hi))
    if not (d0.shape == lo.shape == hi.shape) or np.any(d0 <= lo) or np.any(d0 >= hi):
        raise ValueError("d0, lo and hi need the same shape, with lo < d0 < hi")

    def cost_of(d):
        exp = build(d)
        total = 0.0
        for x in truths:
            F, _ = fisher_information(exp, spec, x, sigma)
            var = jnp.diag(jnp.linalg.inv(F))[idx]
            total = total + jnp.sum(var / jnp.asarray(x)[idx] ** 2)
        return total / len(truths)

    lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi)
    to_d = lambda u: lo_j + (hi_j - lo_j) * jax.nn.sigmoid(u)
    p0 = (d0 - lo) / (hi - lo)
    u = jnp.asarray(np.log(p0 / (1.0 - p0)))
    value_and_grad = jax.jit(jax.value_and_grad(lambda u: cost_of(to_d(u))))

    m, v, b1, b2, eps = jnp.zeros_like(u), jnp.zeros_like(u), 0.9, 0.999, 1e-8
    history = []
    for step in range(n_steps):
        c, g = value_and_grad(u)
        if not (np.isfinite(float(c)) and bool(jnp.all(jnp.isfinite(g)))):
            raise FloatingPointError(f"cost or gradient not finite at step {step}; try float64 or a smaller lr")
        history.append(float(c))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        u = u - lr * (m / (1 - b1 ** (step + 1))) / (jnp.sqrt(v / (1 - b2 ** (step + 1))) + eps)
        if callback is not None:
            callback(step, float(c), np.asarray(to_d(u)))
    design = np.asarray(to_d(u), float)
    history.append(float(value_and_grad(u)[0]))

    def rel(d):
        exp = build(jnp.asarray(d))
        return np.array([[crlb(exp, spec, sigma, x).rel_sd[names.index(t)] for t in targets] for x in truths])
    return DesignResult(design, d0, np.asarray(history), targets, truths, rel(d0), rel(design))
