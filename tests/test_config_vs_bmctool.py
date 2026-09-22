"""The native config reader must agree with bmctool, field for field.

bmctool is the reference implementation of this format. Replacing it with 150 lines of our own
is only safe if the two produce the same JAXParams -- so this compares them on the four configs
shipped with the package, and on a few real ones from the research tree when it is available.

A handful is enough. The format has a small number of branches (rates or times, MT or no MT,
aliases, defaults) and these cover them; running a thousand files would exercise the same
branches a thousand times.

Skipped when bmctool is not installed. A skip is not a pass.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pytest

bmctool_Parameters = None
for _mod, _attr in (("bmctool.parameters.Parameters", "Parameters"),
                    ("cesty.BMC_JAX.bmctool.parameters.Parameters", "Parameters")):
    try:
        import importlib, sys
        root = os.environ.get("DLQCEST_ROOT")
        if "cesty" in _mod:
            if not root:
                continue
            if root not in sys.path:
                sys.path.insert(0, root)
        bmctool_Parameters = getattr(importlib.import_module(_mod), _attr)
        break
    except Exception:
        continue

pytestmark = pytest.mark.skipif(bmctool_Parameters is None, reason="bmctool not importable")

SHIPPED = sorted(str(p) for p in (Path(__file__).parents[1] / "src/bmc_jax/io/configs").glob("*.yaml"))
_root = os.environ.get("DLQCEST_ROOT", "")
RESEARCH = sorted(glob.glob(f"{_root}/cesty/BMC_JAX/data/Sim_cases/case_*/case_*_model.yaml"))[:6] if _root else []


@pytest.mark.parametrize("path", SHIPPED + RESEARCH, ids=lambda p: Path(p).stem)
def test_same_jaxparams_as_bmctool(path):
    from bmc_jax.io import from_yaml
    from bmc_jax.jax_solver import convert_params

    try:
        ref = convert_params(bmctool_Parameters.from_yaml(path), max_cest_pools=8)
    except Exception as exc:
        pytest.skip(f"bmctool rejects this config ({type(exc).__name__}); nothing to compare")
    ours = convert_params(from_yaml(path), max_cest_pools=8)

    assert ref._fields == ours._fields
    for name, a, b in zip(ref._fields, ref, ours):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        if a.dtype.kind in "fi":
            assert np.array_equal(a, b), f"{name}: {a} vs {b}"
