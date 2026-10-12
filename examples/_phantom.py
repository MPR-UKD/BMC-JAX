"""Load the shipped phantom measurement: seven tubes, two ladders, the sequences that produced it.

Tutorials 08 to 11 fit real scanner data rather than a simulation of it. Everything they need
is in `examples/data/` and is described in `examples/data/README.md`; this module is only the
loading, kept in one place so every tutorial reads the same numbers the same way.

The data is a 2026-09-27 phantom scan at 2.8936 T: seven tubes, PBS plus three pairs (GAG at two
concentrations, creatine and carnosine each one solution at two pH), measured once as a QUESP ladder
(eight saturation powers) and once as a QUEST ladder (six saturation times). The sequences are the
ones actually played, converted to events ahead of time: reading Pulseq costs several GB and over a
minute per file, which is why the conversion is shipped and not repeated here.
"""
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from bmc_jax.simulation import SimParams, SimEvent

DATA = Path(__file__).resolve().parent / "data"
FILE = DATA / "phantom_7tubes.npz"

#: Near-water offsets are dropped from both the data and the model. Direct water saturation
#: dominates the least-squares there and carries almost no exchange information, and the
#: single-isochromat forward model is at its least accurate. The fits of record used 1.0 ppm.
EXCLUDE_PPM = 1.0

#: Starting value of the CEST pool's T2 (s). It was not measured for this phantom; the tutorials fit it.
POOL_T2_START = 0.010

#: Arrays with one entry per tube (leading axis), the ones `load_data(tubes=...)` selects from.
PER_TUBE = ("Z_quesp", "Z_quest", "vial_names", "vial_labels", "pH", "vial_dw", "f_nominal", "R1", "R1_start",
            "R1_end", "b0_hz", "rel_b1", "noise_quesp", "noise_quest", "n_voxels")

#: The checks no calibration error shared by two tubes can satisfy: (label, numerator, denominator, expected
#: f ratio). GAG is one stock at 100 and 50 mg/mL; creatine and carnosine are each one solution at two pH.
PAIRS = (("GAG 100 / 50 mg/mL", "GAG100_pH6.31", "GAG50_pH6.20", 2.0),
         ("creatine 7.51 / 6.31", "creatine_pH7.51", "creatine_pH6.31", 1.0),
         ("carnosine 7.44 / 6.59", "carnosine_pH7.44", "carnosine_pH6.59", 1.0))


def load_data(tubes=None):
    """The measurement and every per-tube constant a fit can hold fixed.

    Parameters
    ----------
    tubes : None, "pool" or sequence of int
        None: all seven tubes, PBS first. "pool": the six tubes with a CEST pool (PBS left out). A sequence
        selects those tube indices, in that order.

    Returns
    -------
    dict
        `ppm`, `b1_levels_uT`, `tsat_s`, `Z_quesp` (tubes, 8, 101), `Z_quest` (tubes, 6, 101), `vial_names`,
        `vial_labels`, `pH`, `vial_dw` (NaN: no pool), `f_nominal`, `R1`, `R1_start`, `R1_end`, `b0_hz`,
        `rel_b1`, `noise_quesp`, `noise_quest`, `n_voxels`, `water_r2`, `m_start`, `b0_T`.
    """
    with np.load(FILE, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    if tubes is None:
        return data
    idx = np.flatnonzero(np.isfinite(data["vial_dw"])) if isinstance(tubes, str) and tubes == "pool" \
        else np.asarray(tubes, int)
    return {k: (v[idx] if k in PER_TUBE else v) for k, v in data.items()}


#: Longitudinal magnetisation at the start of saturation, as a fraction of equilibrium. The readout
#: does not let the sample recover fully between offsets, so saturation does not begin at equilibrium.
#: Fitted to the PBS tube alone (QUEST; PBS has no CEST pool, so nothing can trade with it).
M_START = float(load_data()["m_start"])


def load_events(stem, keep=None):
    """One rung's converted sequence.

    Parameters
    ----------
    stem : str
        File stem in `data/events/`, e.g. `quesp_sim_b1_1uT` or `quest_sim_tsat_2s`.
    keep : ndarray of bool, optional
        Offset mask to apply. Must be the same mask applied to the measurement: `events` is
        indexed by offset on its leading axis, so masking one without the other compares offset
        i of the data against offset j of the model, silently.

    Returns
    -------
    SimEvent
        Batched events, shape (n_offsets, n_events).

    Notes
    -----
    The sequences acquire an unsaturated reference at -300 ppm as their first offset, so a file
    carries one row more than the Z-spectrum has points. That row is dropped here.
    """
    with np.load(DATA / "events" / f"{stem}_events.npz") as d:
        arrays = {f: np.asarray(d[f]) for f in SimEvent._fields}
        n_offsets = len(d["offsets_ppm"])
    ev = SimEvent(**{f: jnp.asarray(a[1:] if len(a) == n_offsets else a,
                                    jnp.int32 if f in ("type", "adc_idx") else jnp.float32)
                     for f, a in arrays.items()})
    return ev if keep is None else __import__("jax").tree.map(lambda a: a[keep], ev)


def quesp_stem(b1_uT):
    """Event file stem of the QUESP rung at `b1_uT`."""
    return f"quesp_sim_b1_{('%g' % b1_uT).replace('.', 'p')}uT"


def quest_stem(tsat_s):
    """Event file stem of the QUEST rung at `tsat_s`."""
    return f"quest_sim_tsat_{('%g' % tsat_s).replace('.', 'p')}s"


def vial_params(data, pool_t2_s=POOL_T2_START):
    """The tubes as one batched `SimParams`, with everything the fit does not vary.

    Each tube gets its own measured water R1, B0 offset and relative B1; water R2 is the PBS value for
    every tube. A tube with a CEST pool gets it at its chemical shift with T2 `pool_t2_s`; exchange rate
    and pool size are placeholders, they are what the fit solves for. A tube without one (PBS) gets a
    pool of size 0 and rate 0, which leaves its water exactly as without the pool.

    Parameters
    ----------
    data : dict
        From `load_data`.
    pool_t2_s : float
        T2 of the CEST pool (s), the start of a fit that frees it.

    Returns
    -------
    SimParams
        Leading axis of one entry per tube, in the order of `data["vial_names"]`.
    """
    n = len(data["vial_names"])
    ones = np.ones(n, np.float32)
    pool = np.isfinite(data["vial_dw"])
    return SimParams(
        water_r1=jnp.asarray(data["R1"]),
        water_r2=jnp.asarray(float(data["water_r2"]) * ones),
        water_f=jnp.asarray(ones),
        cest_k=jnp.asarray(np.where(pool, 900.0, 0.0).astype(np.float32)[:, None]),
        cest_f=jnp.asarray(np.where(pool, 0.008, 0.0).astype(np.float32)[:, None]),
        cest_dw=jnp.asarray(np.where(pool, data["vial_dw"], 0.0).astype(np.float32)[:, None]),
        cest_r1=jnp.asarray(ones[:, None]),
        cest_r2=jnp.asarray((ones / pool_t2_s)[:, None]),
        mt_k=jnp.asarray(0 * ones), mt_f=jnp.asarray(0 * ones), mt_dw=jnp.asarray(0 * ones),
        mt_r1=jnp.asarray(0 * ones), mt_r2=jnp.asarray(ones),
        mt_lineshape=jnp.zeros(n, jnp.int32),
        b0=jnp.asarray(float(data["b0_T"]) * ones),
        gamma=jnp.asarray(267.5153 * ones),
        b0_inhom=jnp.asarray(data["b0_hz"] / (42.5764 * float(data["b0_T"]))),
        rel_b1=jnp.asarray(data["rel_b1"]))


def start_magnetisation(params, m_start=M_START):
    """Per-tube starting magnetisation at `m_start`, stacked to match `vial_params`."""
    import jax

    from bmc_jax.simulation import initial_state

    n = params.water_r1.shape[0]
    return jnp.stack([initial_state(
        jax.tree.map(lambda a, i=i: a[i], params), m_start=m_start) for i in range(n)])


def pair_ratios(f, names):
    """The pair checks for one set of pool sizes: [(label, f ratio, expected)], tubes missing from `names` skipped."""
    names = [str(n) for n in names]
    return [(label, float(f[names.index(a)] / f[names.index(b)]), want)
            for label, a, b, want in PAIRS if a in names and b in names]


def print_pairs(f, names, title="pair checks"):
    """Print `pair_ratios`: the fitted f ratio against what the recipe says it must be."""
    print(f"{title} (f ratio; a shared calibration error cancels in a ratio):")
    for label, r, want in pair_ratios(f, names):
        print(f"  {label:<22} {r:6.3f}   (must be {want:.0f})")
