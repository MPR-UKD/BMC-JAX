"""The forward model must be BIT-IDENTICAL to the copy that produced the phantom results.

WHY THIS TEST EXISTS
  This package was lifted out of a research tree in which five copies of the same solver had
  drifted apart -- one of them, the copy CLAUDE.md names as canonical, still hardcodes
  `dtype=jnp.float64` in get_initial_magnetization where the others honour the run's precision.
  A reorganisation that silently changes a number is worse than no reorganisation, because every
  result already quoted from the old copy would need re-deriving without anyone knowing.

  So the tolerance here is ZERO, not 1e-6. Both paths run the same arithmetic in the same order
  at the same precision; if the restructuring is faithful the outputs are identical bit for bit,
  and any difference at all is a real bug rather than numerical noise.

  This is the gate every later step has to pass: renaming jax_solver -> solvers, splitting
  fitting.py into a package, making the optimiser an argument. Run it before and after each.

REQUIREMENTS
  The DLQCest research tree, for the reference copy and its cached forward models. Skipped when
  it is not present, so the package still tests standalone -- but a skip is not a pass, and the
  reorganisation must not be merged on a skip.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

DLQ = Path(os.environ.get("DLQCEST_ROOT", "/home/gizmoo/DLQCest"))
EVAL = DLQ / "GPU_fitting" / "evaluation"
REF = DLQ / "GPU_fitting" / "fitting_app" / "cesty" / "BMC_JAX"

pytestmark = pytest.mark.skipif(
    not (REF / "jax_solver" / "simulator.py").exists(),
    reason=f"reference tree not found at {REF}; set DLQCEST_ROOT")


def test_spectrum_is_bit_identical():
    os.environ.setdefault("FIT_INPUTS_DIR", str(EVAL / "res2009" / "fit" / "inputs"))
    os.environ.setdefault("FIT_INPUTS_NAME", "fit_inputs_b0corr.npz")
    os.environ.setdefault("FIT_CEST_T2_PERVIAL", "1")
    for p in (EVAL / "fit", REF, REF / "scripts" / "lib"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import jax
    from prepare_events import load_events
    from fit_vials import build_base, start_magnetization
    import jax_solver.simulator as reference
    import bmc_jax.jax_solver.simulator as package

    assert "BMC-JAX" in package.__file__, "the installed package was shadowed by a sys.path copy"

    ev, meta = load_events("quesp_sim_b1_1uT")
    if meta["n_batches"] == 102:
        ev = jax.tree.map(lambda a: a[1:], ev)       # drop the -300 ppm M0 event

    # One vial's fixed calibration, taken from the fit of record: GAG at 1 uT.
    base = build_base(1, np.float32([0.348]), np.float32([0.613]),
                      np.float32([0.0]), np.float32([0.996]), 1.0)
    p = jax.tree.map(lambda a: a[0], base)
    m0 = start_magnetization(p, scale=0.582)
    water_z = 4                                      # 2 * (1 + n_pools), n_pools = 1

    za = np.asarray(reference.simulate_spectrum(p, ev, m0, return_full_trajectory=False)[:, water_z])
    zb = np.asarray(package.simulate_spectrum(p, ev, m0, return_full_trajectory=False)[:, water_z])

    assert za.shape == zb.shape
    assert 0.0 < za.min() < za.max() < 1.05, f"reference Z is not physical: {za.min()}..{za.max()}"
    assert np.abs(za - zb).max() == 0.0, (
        f"the package is no longer bit-identical to the reference: "
        f"max |dZ| = {np.abs(za - zb).max():.3e}")
