"""Protocol design: how precisely a protocol can measure a parameter, and a protocol that does better.

    crlb.py      the Cramer-Rao lower bound of the parameters of a FitSpec   (crlb, CRLB, fisher_information)
    optimize.py  a design with a lower bound, by gradient descent             (optimize_design, DesignResult)

Compute bounds in float64 (bmc_jax.settings.use_x64): the Fisher matrix of an exchange rate and a pool
size is close to singular.
"""
from .crlb import CRLB, crlb, fisher_information, model_of
from .optimize import DesignResult, optimize_design

__all__ = ["crlb", "CRLB", "fisher_information", "model_of", "optimize_design", "DesignResult"]
