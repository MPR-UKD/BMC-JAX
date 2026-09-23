"""The whole fitting chain must reproduce a real measured result, not just the forward model.

test_equivalence_solver.py checks one simulated spectrum. This checks the thing that matters: a
joint fit of exchange rate and pool size across a seven-rung B1 ladder, on real scanner data,
against the numbers the research pipeline produced and that were quoted from.

Every fixed input is the one that fit used -- the same B0-corrected Z, the same cached forward
models, the same gap spoiling, per-vial pool T2, M_start 0.582, PBS water R2 for every tube,
per-vial R1 and rel_B1, and the same near-water mask. The only thing that changes is which code
runs the fit.

THE CHECK THAT NEEDS NO CALIBRATION
  The two creatine tubes are the same solution at the same concentration, so their fitted pool
  sizes must come out EQUAL whatever the absolute calibration is. That ratio is asserted
  separately, because it is the one number here that cannot be fooled by a systematic error
  common to both.

Requires DLQCEST_ROOT. Skipped without it, and a skip is not a pass.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

DLQ = Path(os.environ["DLQCEST_ROOT"]) if os.environ.get("DLQCEST_ROOT") else None
EVAL = DLQ / "GPU_fitting" / "evaluation" if DLQ else None
BUNDLE = EVAL / "res2009" / "fit" / "inputs" / "fit_inputs_b0corr.npz" if EVAL else None

pytestmark = pytest.mark.skipif(
    BUNDLE is None or not BUNDLE.exists(),
    reason="set DLQCEST_ROOT to the research tree to run the phantom gate")

#: The fit of record, 2026-09-21 quesp-joint-t2pv-flo1e-5: (k [Hz], f / f_expected).
RECORD = {"creatine_pH6.25": (104.5, 1.009), "creatine_pH7.87": (249.2, 0.977),
          "glucose_pH6.50": (585.2, 1.204), "GAG_pH6.64": (794.1, 0.772)}
PAIR_RATIO = 1.032


def _fit_the_phantom():
    import jax
    import jax.numpy as jnp

    os.environ.update(FIT_INPUTS_DIR=str(BUNDLE.parent), FIT_INPUTS_NAME=BUNDLE.name,
                      FIT_CEST_T2_PERVIAL="1")
    for p in (EVAL / "fit", DLQ / "GPU_fitting/fitting_app/cesty/BMC_JAX",
              DLQ / "GPU_fitting/fitting_app/cesty/BMC_JAX/scripts/lib"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from prepare_events import load_events
    from fit_vials import spoil_gaps, EXCLUDE_PPM, cest_t2_for

    from bmc_jax.jax_solver import JAXParams, get_initial_magnetization
    from bmc_jax.fitting import Experiment, FitSpec, Slot, fit

    d = np.load(    BUNDLE, allow_pickle=True)
    ppm, labels, levels = d["ppm"], d["labels"], d["b1_levels_uT"]
    keep = np.abs(ppm) >= EXCLUDE_PPM
    PBS = 1
    r2_pbs = float(np.median(d["R2"].reshape(-1)[(labels == PBS).reshape(-1)]))

    # ---- events: same cache, same gap spoiling, concatenated over the ladder --------------------
    ev_list = []
    for b1 in levels:
        stem = f"quesp_sim_b1_{('%g' % b1).replace('.', 'p')}uT"
        ev, meta = load_events(stem)
        if meta["n_batches"] == ppm.size + 1:
            ev = jax.tree.map(lambda a: a[1:], ev)
        ev, _ = spoil_gaps(ev)
        ev_list.append(jax.tree.map(lambda a: a[keep], ev))
    events = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), *ev_list)

    # ---- one row per vial, exactly the fixed inputs of the fit of record ------------------------
    vials = [2, 3, 4, 5]
    ids, names, dws = list(d["vial_ids"]), list(d["vial_names"]), d["vial_dw"]
    rows, Z = [], []
    for v in vials:
        j = ids.index(v); m = (labels == v).reshape(-1)
        rows.append(dict(name=str(names[j]), dw=float(dws[j]),
                         r1=float(np.median(d["R1"].reshape(-1)[m])),
                         b0=float(np.median(d["b0_hz"].reshape(-1)[m])),
                         rb1=float(np.median(d["rel_b1"].reshape(-1)[m]))))
        Z.append(np.concatenate([d["Z"][k].reshape(ppm.size, -1).T[m].mean(0)[keep]
                                 for k in range(len(levels))]))
    t2_pool, _ = cest_t2_for([r["name"] for r in rows])
    n = len(rows); o = np.ones(n, np.float32)
    base = JAXParams(
        water_r1=jnp.asarray([r["r1"] for r in rows], jnp.float32),
        water_r2=jnp.asarray(r2_pbs * o), water_f=jnp.asarray(o),
        cest_k=jnp.asarray((900.0 * o)[:, None]), cest_f=jnp.asarray((0.008 * o)[:, None]),
        cest_dw=jnp.asarray(np.array([[r["dw"]] for r in rows], np.float32)),
        cest_r1=jnp.asarray((1.0 * o)[:, None]), cest_r2=jnp.asarray((1.0 / t2_pool)[:, None]),
        mt_k=jnp.asarray(0*o), mt_f=jnp.asarray(0*o), mt_dw=jnp.asarray(0*o),
        mt_r1=jnp.asarray(0*o), mt_r2=jnp.asarray(1.0*o), mt_lineshape=jnp.zeros(n, jnp.int32),
        b0=jnp.asarray(2.8936*o), gamma=jnp.asarray(267.5153*o),
        b0_inhom=jnp.asarray(np.array([r["b0"] for r in rows], np.float32) / 123.2),
        rel_b1=jnp.asarray([r["rb1"] for r in rows], jnp.float32),
        use_pulse_frame=jnp.zeros(n, jnp.int32))
    m0 = jnp.stack([get_initial_magnetization(jax.tree.map(lambda a, i=i: a[i], base), scale=0.582)
                    for i in range(n)])

    exp = Experiment(base, events, m0, 4)
    spec = FitSpec([Slot("k", "cest_k", 0, scale=1000.0, lo=5.0, hi=5000.0, x0=300.0),
                    Slot("f", "cest_f", 0, scale=0.01, lo=1e-5, hi=0.03, x0=0.008)])
    res = fit(exp, spec, np.stack(Z), n_iter=20, n_per_block=int(keep.sum()))
    return rows, res, {str(nm): float(f) for nm, f in zip(d["vial_names"], d["f_expected"])}


@pytest.fixture(scope="module")
def phantom():
    return _fit_the_phantom()


@pytest.mark.parametrize("vial", sorted(RECORD))
def test_exchange_rate_matches_the_fit_of_record(phantom, vial):
    rows, res, _ = phantom
    i = [r["name"] for r in rows].index(vial)
    k_ref = RECORD[vial][0]
    assert res["k"][i] == pytest.approx(k_ref, rel=2e-3), (
        f"{vial}: k = {res['k'][i]:.1f} against a record of {k_ref}")


@pytest.mark.parametrize("vial", sorted(RECORD))
def test_pool_size_matches_the_fit_of_record(phantom, vial):
    rows, res, fexp = phantom
    i = [r["name"] for r in rows].index(vial)
    assert res["f"][i] / fexp[vial] == pytest.approx(RECORD[vial][1], abs=2e-3)


def test_the_creatine_pair_agrees(phantom):
    """Same solution, same concentration: the two fitted pool sizes must be equal.

    This is the one assertion here that no common systematic error can satisfy by accident, so
    it is worth more than any single absolute number above.
    """
    rows, res, _ = phantom
    names = [r["name"] for r in rows]
    ratio = res["f"][names.index("creatine_pH6.25")] / res["f"][names.index("creatine_pH7.87")]
    assert ratio == pytest.approx(PAIR_RATIO, abs=2e-3)
