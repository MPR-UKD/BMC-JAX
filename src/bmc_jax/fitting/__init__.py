"""Fit selected parameters of an Experiment to measured Z-spectra.

    experiment.py   WHAT is simulated                (Experiment, Experiment.stack)
    params.py       WHICH parameters are free        (FreeParam, FitSpec, write_params)
    residual.py     WHAT is compared                 (residual_one, residual_batch)
    lm.py           HOW it is minimised              (levenberg_marquardt)
    independent.py  every sample on its own          (fit, fit_one, fit_batch, fit_batch_sharded, fit_batch_adam)
    joint.py        samples together                 (fit_joint: shared values, values free in some samples only)
    result.py       WHAT comes back                  (FitResult)

One-directional imports, and one rule worth stating out loud: `lm.py` must not import the simulation.
A bounded Levenberg-Marquardt has nothing to do with Bloch-McConnell, and keeping that boundary is
what lets `levenberg_marquardt` be used on its own.

Nothing here knows about any particular agent. A FitSpec names SimParams fields; which pools exist
and what they are called belongs to the model.

Renamed on 2026-10-07 (old -> new table in the README): Slot -> FreeParam, apply_x -> write_params,
lm_minimize -> levenberg_marquardt, make_single_residual / make_phantom_residual -> residual_one /
residual_batch, fit_voxel -> fit_one, fit_phantom_lm(_sharded) / fit_phantom_adam -> fit_batch(_sharded) /
fit_batch_adam; pad_events lives in bmc_jax.simulation.
"""
from .experiment import Experiment
from .params import FitSpec, FreeParam, write_params
from .residual import residual_one, residual_batch
from .lm import levenberg_marquardt
from .independent import fit, fit_one, fit_batch, fit_batch_sharded, fit_batch_adam
from .joint import fit_joint
from .result import FitResult

__all__ = [
    "Experiment",                                      # what is simulated
    "FreeParam", "FitSpec", "write_params",            # what is free
    "residual_one", "residual_batch",
    "levenberg_marquardt",                             # bounded LM, usable standalone
    "fit", "FitResult",                                # the entry point to prefer
    "fit_joint",                                       # shared / partly free parameters
    "fit_one", "fit_batch", "fit_batch_sharded", "fit_batch_adam",
]
