"""Load the shipped phantom measurement: four vials, two ladders, the sequences that produced it.

Tutorials 08 and 09 fit real scanner data rather than a simulation of it. Everything they need
is in `examples/data/` and is described in `examples/data/README.md`; this module is only the
loading, kept in one place so both tutorials read the same numbers the same way.

The data is a 2026-09-20 phantom scan at 2.8936 T: four tubes of known concentration, measured
once as a QUESP ladder (seven saturation powers) and once as a QUEST ladder (six saturation
times). The sequences are the ones actually played, converted to events ahead of time -- reading
Pulseq costs several GB and over a minute per file, which is why the conversion is shipped and
not repeated here.
"""
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from bmc_jax.jax_solver import JAXParams, SimEvent

DATA = Path(__file__).resolve().parent / "data"

#: Near-water offsets are dropped from both the data and the model. Direct water saturation
#: dominates the least-squares there and carries almost no exchange information, and the
#: single-isochromat forward model is at its least accurate. The fit of record used 1.0 ppm.
EXCLUDE_PPM = 1.0

#: Longitudinal magnetisation at the start of saturation, as a fraction of equilibrium. The
#: readout does not let the sample recover fully between offsets, so saturation does not begin
#: at equilibrium. Measured for this phantom.
M_START = 0.582


def load_data():
    """The measurement and every per-vial constant the fit holds fixed.

    Returns
    -------
    dict
        `ppm`, `b1_levels_uT`, `tsat_s`, `Z_quesp` (4, 7, 101), `Z_quest` (4, 6, 101),
        `vial_names`, `vial_dw`, `R1`, `b0_hz`, `rel_b1`, `f_expected`, `cest_t2_s`,
        `water_r2`, `m_start`, `b0_T`.
    """
    with np.load(DATA / "phantom_4vials.npz", allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


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


def vial_params(data):
    """The four vials as one batched `JAXParams`, with everything the fit does not vary.

    Each vial gets its own measured water R1, B0 offset and relative B1, and its own CEST-pool
    T2; water R2 is the PBS value for every tube. Exchange rate and pool size are left at
    placeholders -- they are what the fit solves for.

    Parameters
    ----------
    data : dict
        From `load_data`.

    Returns
    -------
    JAXParams
        Leading axis of length 4, in the order of `data["vial_names"]`.
    """
    n = len(data["vial_names"])
    ones = np.ones(n, np.float32)
    return JAXParams(
        water_r1=jnp.asarray(data["R1"]),
        water_r2=jnp.asarray(float(data["water_r2"]) * ones),
        water_f=jnp.asarray(ones),
        cest_k=jnp.asarray((900.0 * ones)[:, None]),
        cest_f=jnp.asarray((0.008 * ones)[:, None]),
        cest_dw=jnp.asarray(data["vial_dw"][:, None]),
        cest_r1=jnp.asarray(ones[:, None]),
        cest_r2=jnp.asarray((1.0 / data["cest_t2_s"])[:, None]),
        mt_k=jnp.asarray(0 * ones), mt_f=jnp.asarray(0 * ones), mt_dw=jnp.asarray(0 * ones),
        mt_r1=jnp.asarray(0 * ones), mt_r2=jnp.asarray(ones),
        mt_lineshape=jnp.zeros(n, jnp.int32),
        b0=jnp.asarray(float(data["b0_T"]) * ones),
        gamma=jnp.asarray(267.5153 * ones),
        b0_inhom=jnp.asarray(data["b0_hz"] / (42.5764 * float(data["b0_T"]))),
        rel_b1=jnp.asarray(data["rel_b1"]),
        use_pulse_frame=jnp.zeros(n, jnp.int32))


def start_magnetisation(params):
    """Per-vial starting magnetisation at `M_START`, stacked to match `vial_params`."""
    import jax

    from bmc_jax.jax_solver import get_initial_magnetization

    n = params.water_r1.shape[0]
    return jnp.stack([get_initial_magnetization(
        jax.tree.map(lambda a, i=i: a[i], params), scale=M_START) for i in range(n)])
