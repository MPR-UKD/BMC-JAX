"""crlb.py -- the Cramer-Rao lower bound: the smallest uncertainty a protocol allows.

For independent Gaussian noise of standard deviation sigma on every measured point, the Fisher
information of the parameters x is

    F = J^T J / sigma^2,      J = dZ/dx, the Jacobian of the model (points x parameters).

No unbiased estimate of x can have a smaller covariance than F^-1, the Cramer-Rao lower bound. Its
diagonal gives the smallest possible standard deviation of each parameter, and its off-diagonal
elements their correlation. The bound is local: it is computed at given true values and changes with
them. It assumes the model is correct and contains no systematic error.

The model is the one `fit` inverts: the FitSpec's parameters written into the experiment's params
(`write_params`), the start state held fixed, water Mz returned. Plain values per sample only.

Precision: the Fisher matrix of an exchange rate and a pool size is close to singular, and float32
rounding in the Jacobian can change its inverse by a large factor. Compute bounds in float64
(`bmc_jax.settings.use_x64(True)`).
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..fitting.params import write_params
from ..simulation.simulate import simulate_spectrum


def _check(experiment, spec):
    if jnp.asarray(experiment.params.water_r1).ndim != 0:
        raise ValueError("the bound is computed for one sample: give an Experiment whose params carry one sample")
    if not spec.independent:
        raise ValueError("the bound supports plain values per sample (FreeParam scope='sample', mode='value', no "
                         "samples=, no m_start or b0_spread)")


def model_of(experiment, spec):
    """The model as a function of the parameters in physical units, x -> Z (water Mz at every offset)."""
    _check(experiment, spec)
    base, events, m0, mz = experiment.params, experiment.events, experiment.m0, experiment.mz_loc
    scales = jnp.asarray(spec.scales, base.water_r1.dtype)

    def z(x):
        p = write_params(base, x / scales, spec)
        return simulate_spectrum(p, events, m0, return_full_trajectory=False)[:, mz]
    return z


def fisher_information(experiment, spec, x, sigma):
    """Fisher information matrix F = J^T J / sigma^2 at x (physical units), and the Jacobian J.

    Pure JAX, so it can itself be differentiated, for example with respect to the protocol
    (bmc_jax.design.optimize_design).
    """
    x = jnp.asarray(x, experiment.params.water_r1.dtype)
    J = jax.jacfwd(model_of(experiment, spec))(x)
    return J.T @ J / sigma ** 2, J


@dataclass(frozen=True)
class CRLB:
    """The bound at the true values `x`. Attributes are NumPy arrays in physical units."""
    names: tuple
    x: np.ndarray            # the true values at which the bound was computed
    sd: np.ndarray           # the smallest possible standard deviation of each parameter
    corr: np.ndarray         # correlation matrix of the estimates
    fisher: np.ndarray       # Fisher information matrix
    jacobian: np.ndarray     # dZ/dx, (points, parameters)

    @property
    def rel_sd(self) -> np.ndarray:
        """The bound relative to the true value, sd / |x|."""
        return self.sd / np.abs(self.x)

    def __getitem__(self, name: str) -> float:
        """The bound (standard deviation) of one parameter, by its FreeParam name."""
        return float(self.sd[self.names.index(name)])

    def summary(self) -> str:
        """A table of the true values, the bound and the relative bound."""
        rows = [f"{'parameter':>10} {'value':>12} {'CRLB (SD)':>12} {'relative':>9}"]
        for n, x, s, r in zip(self.names, self.x, self.sd, self.rel_sd):
            rows.append(f"{n:>10} {x:>12.6g} {s:>12.4g} {100 * r:>8.2f}%")
        return "\n".join(rows)


def crlb(experiment, spec, sigma, x=None):
    """The Cramer-Rao lower bound of the parameters in `spec` for one sample.

    Parameters
    ----------
    experiment
        An :class:`~bmc_jax.fitting.experiment.Experiment` for one sample: the protocol (its events)
        and every parameter that is not in `spec`.
    spec
        The parameters to be estimated (plain values per sample), as for `fit`.
    sigma
        Standard deviation of the noise on every measured point, in the units of Z (0.01 = 1 % of M0).
    x
        The true values, physical units, one per parameter of `spec`. Default: each FreeParam's x0.

    Returns
    -------
    CRLB
        `sd` (the bound), `rel_sd`, `corr`, `fisher` and `jacobian`; `summary()` prints a table.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> b = crlb(Experiment(params, ladder_events, m0, mz_loc), FitSpec([K, F]), sigma=0.002, x=[150.0, 9e-4])
    >>> b["k"], b.rel_sd, b.corr[0, 1]
    """
    x = np.asarray(spec.x0_phys if x is None else x, float)
    F, J = fisher_information(experiment, spec, x, sigma)
    F = np.asarray(F, np.float64)
    cov = np.linalg.inv(F)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = cov / np.outer(sd, sd)
    return CRLB(tuple(s.name for s in spec.slots), x, sd, corr, F, np.asarray(J))
