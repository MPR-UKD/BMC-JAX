"""params.py (formerly spec.py) -- declare WHICH parameters are free, and write a solution back into them.

This is the whole configuration surface of the fitter. Nothing here knows about GAG, amide,
NOE or any other agent: a FreeParam names a SimParams field and a pool index, and everything else --
which pools exist, how many, what they are called -- lives in the model, not in the fitter.

`scale` matters more than it looks. The optimiser works in SCALED units so that every parameter
is O(1); k ~ 300 /s and f ~ 0.008 differ by five orders of magnitude, and an optimiser stepping
on the raw values would take steps that are enormous in one and invisible in the other.

Per-voxel B0 and B1 are fitted simply by listing "b0_inhom" and "rel_b1" as slots. They are real
per-voxel parameters, not something baked into the sequence.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import jax.numpy as jnp

from ..simulation.params import SimParams

@dataclass(frozen=True)
class FreeParam:
    name: str           # human label, e.g. "gag_k"
    field: str          # SimParams field, e.g. "cest_k", "cest_f", "b0_inhom", "rel_b1"
    index: Optional[int] # pool index for array fields (cest_*); None for scalar fields
    scale: float        # rough magnitude, so the optimiser sees O(1) numbers
    lo: float           # lower bound (physical units)
    hi: float           # upper bound (physical units)
    x0: float           # initial guess (physical units)


@dataclass(frozen=True)
class FitSpec:
    slots: List[FreeParam]

    @property
    def n(self):                 return len(self.slots)
    @property
    def scales(self):            return jnp.array([s.scale for s in self.slots])
    @property
    def lo_phys(self):           return np.array([s.lo for s in self.slots])
    @property
    def hi_phys(self):           return np.array([s.hi for s in self.slots])
    @property
    def x0_phys(self):           return np.array([s.x0 for s in self.slots])
    # scaled versions (what the optimiser actually works in)
    def to_scaled(self, x_phys): return np.asarray(x_phys) / np.array([s.scale for s in self.slots])
    def to_phys(self, x_scaled): return np.asarray(x_scaled) * np.array([s.scale for s in self.slots])


def write_params(base: SimParams, x_scaled: jnp.ndarray, spec: FitSpec) -> SimParams:
    """x_scaled -> physical -> written into the named slots. Pure JAX; safe under vmap."""
    dt = base.water_r1.dtype                      # match the params' dtype (fp32/fp64)
    x_phys = (x_scaled * spec.scales).astype(dt)
    updates = {}
    for i, s in enumerate(spec.slots):
        val = x_phys[i]
        if s.index is None:
            updates[s.field] = val
        else:
            cur = updates.get(s.field, getattr(base, s.field))
            updates[s.field] = cur.at[s.index].set(val)
    return base._replace(**updates)
