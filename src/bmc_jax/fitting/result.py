"""What a fit returns: the values, the fitted curves, and enough to judge them.

A bare array of numbers is not a result. Every caller ends up recomputing the same three things
from it -- the residual, which parameters ran into their bounds, and how the error is distributed
across the experiment -- and each of them is a chance to compute it slightly differently. They
belong with the numbers.

`at_bounds` is not a convenience. A parameter sitting on its bound is not a measurement: the
optimiser wanted to go further and was not allowed to, so the value reported is the bound, not
the data's answer. That has to be visible without anyone having to look for it.

`fit_joint` fills in more: which values were free in which sample (`free`), error bars and correlations
(`sd`, `corr`), the starting values (`x0`, so `unmoved` can say which values never left them) and the
cost after every step. `fit` leaves them None.
"""
from dataclasses import dataclass

import numpy as np

from .params import FitSpec


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters and everything needed to decide whether to believe them.

    Attributes
    ----------
    x
        Fitted values in physical units, shape (n_samples, n_parameters).
    z_fit
        The model evaluated at `x`, same shape as `z_meas`.
    z_meas
        The data that was fitted.
    spec
        The `FitSpec` used, which names the parameters and carries their bounds.
    n_per_block
        Offsets per experiment when several were stacked, so the residual can be reported per
        rung of a ladder. None when the fit was a single experiment.
    free
        (fit_joint) True where the fit set the value; elsewhere `x` is the value the sample's params
        carry. None (fit): every value was fitted.
    sd
        (fit_joint) Standard deviations, physical units, NaN where not free. Scaled by the misfit
        (s^2 = cost / (points - free values)): when the model and not the noise limits the fit, they
        say how well the data pin a value down GIVEN the model, not how right the model is.
    corr
        (fit_joint) Correlations between one sample's free values, shape (n_samples, n_params,
        n_params), NaN where not free. A shared value appears in every sample it is free in.
    x0
        (fit_joint) Starting values, physical units.
    cost
        (fit_joint) Sum of squared residuals after every LM step.
    nan_steps
        (fit_joint) LM steps rejected because the Jacobian had NaN / inf.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> res = fit(experiment, spec, z)
    >>> res["k"]
    array([104.5, 249.2])
    >>> res.at_bounds.any()
    False
    """

    x: np.ndarray
    z_fit: np.ndarray
    z_meas: np.ndarray
    spec: FitSpec
    n_per_block: int | None = None
    free: np.ndarray | None = None
    sd: np.ndarray | None = None
    corr: np.ndarray | None = None
    x0: np.ndarray | None = None
    cost: np.ndarray | None = None
    nan_steps: int = 0

    @property
    def names(self) -> list[str]:
        """Parameter names, in the order they appear in `x`."""
        return [s.name for s in self.spec.slots]

    def __getitem__(self, name: str) -> np.ndarray:
        """Fitted values of one parameter across all samples, by name.

        Raises
        ------
        KeyError
            If no slot has that name.
        """
        try:
            return self.x[:, self.names.index(name)]
        except ValueError:
            raise KeyError(f"no parameter {name!r}; the fit has {self.names}") from None

    @property
    def residual(self) -> np.ndarray:
        """Model minus data, same shape as `z_meas`."""
        return self.z_fit - self.z_meas

    @property
    def rms(self) -> np.ndarray:
        """Root-mean-square residual per sample."""
        return np.sqrt(np.mean(self.residual ** 2, axis=-1))

    @property
    def rms_per_block(self) -> np.ndarray | None:
        """RMS per sample per experiment, shape (n_samples, n_blocks), or None.

        Worth looking at whenever several experiments were stacked. A model error that grows with
        saturation power is invisible in the pooled RMS and obvious here -- the pooled number
        averages it away with the rungs that fit well.
        """
        if self.n_per_block is None:
            return None
        n_blocks = self.z_meas.shape[-1] // self.n_per_block
        r = self.residual[..., :n_blocks * self.n_per_block]
        return np.sqrt(np.mean(r.reshape(*r.shape[:-1], n_blocks, self.n_per_block) ** 2, axis=-1))

    @property
    def at_bounds(self) -> np.ndarray:
        """Boolean mask, shape like `x`: True where a value sits on its bound.

        A True here means that parameter was not measured. The optimiser wanted to leave the
        allowed range and could not, so the number reported is where the bound is.
        """
        lo, hi = np.asarray(self.spec.lo_phys), np.asarray(self.spec.hi_phys)
        railed = np.isclose(self.x, lo) | np.isclose(self.x, hi)
        return railed if self.free is None else railed & self.free

    @property
    def unmoved(self) -> np.ndarray | None:
        """Boolean mask like `x`: a free value still exactly at its start, or None without `x0`.

        A value that never left its start accepted no step at all. With a NaN Jacobian or a start in a
        flat corner, the number reported is the start, not a fit -- worth knowing before reading it.
        """
        if self.x0 is None:
            return None
        still = np.isclose(self.x, self.x0, rtol=1e-6, atol=0.0)
        return still if self.free is None else still & self.free

    def summary(self) -> str:
        """A table of the fit, one row per sample, with railed values marked."""
        head = f"{'sample':>6}  " + "  ".join(f"{n:>12}" for n in self.names) + f"  {'rms':>9}"
        rows = [head, "-" * len(head)]
        for i in range(self.x.shape[0]):
            cells = []
            for j, _ in enumerate(self.names):
                mark = "*" if self.at_bounds[i, j] else ("=" if self.free is not None and not self.free[i, j]
                                                        else " ")
                cells.append(f"{self.x[i, j]:>11.5g}{mark}")
            rows.append(f"{i:>6}  " + "  ".join(cells) + f"  {self.rms[i]:>9.2e}")
        if self.at_bounds.any():
            rows.append("* on a bound -- not a measurement, the optimiser was stopped there")
        if self.free is not None and not self.free.all():
            rows.append("= not free in this sample: the value its params carry")
        return "\n".join(rows)

    def __repr__(self) -> str:
        railed = int(self.at_bounds.sum())
        return (f"FitResult({self.x.shape[0]} samples, {self.names}, "
                f"rms {self.rms.mean():.3e}" + (f", {railed} at bounds)" if railed else ")"))
