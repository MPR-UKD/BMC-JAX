"""Fitting: choose the free parameters, build a residual, minimise it.

    spec.py       WHICH parameters are free           (Slot, FitSpec, apply_x)
    residual.py   WHAT is compared                    (make_*_residual)
    optimize.py   HOW it is minimised                 (lm_minimize)
    batching.py   over WHAT it is parallelised        (fit_voxel, fit_phantom_*)

One-directional imports, and one rule worth stating out loud: `optimize.py` must not import the
solver. A bounded Levenberg-Marquardt has nothing to do with Bloch-McConnell, and keeping that
boundary is what lets `lm_minimize` be used on its own.

Nothing here knows about any particular agent. A FitSpec names JAXParams fields; which pools
exist and what they are called belongs to the model.

This was one 375-line module. The split moved no logic -- every function was lifted verbatim --
and every name it exported is still exported here.
"""
from .spec import FitSpec, Slot, apply_x
from .residual import make_single_residual, make_phantom_residual, _cast_problem
from .optimize import lm_minimize, _lm_one, c_out_select
from .batching import fit_voxel, fit_phantom_lm, fit_phantom_lm_sharded, fit_phantom_adam

__all__ = [
    "Slot", "FitSpec", "apply_x",                 # what is free
    "make_single_residual", "make_phantom_residual",
    "lm_minimize",                                # bounded LM, usable standalone
    "fit_voxel", "fit_phantom_lm", "fit_phantom_lm_sharded", "fit_phantom_adam",
]
