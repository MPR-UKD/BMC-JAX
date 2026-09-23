# Shipped phantom measurement

Tutorials 08 and 09 fit real scanner data. This is that data, and the sequences that produced it.

## The scan

A four-tube phantom measured at 2.8936 T on 2026-09-20. Each tube is a solution of known
concentration, so the pool size the fit recovers can be checked against what was weighed in.

| tube | pool offset | notes |
|---|---|---|
| `creatine_pH6.25` | 1.9 ppm | guanidinium |
| `creatine_pH7.87` | 1.9 ppm | same solution, same concentration, higher pH |
| `glucose_pH6.50` | 1.2 ppm | hydroxyl |
| `GAG_pH6.64` | 1.0 ppm | hydroxyl |

The two creatine tubes are the same solution at the same concentration. Their fitted pool sizes
must therefore come out equal, whatever the absolute calibration is — the one check here that no
systematic error shared by both tubes can satisfy.

The same tubes were measured twice: a QUESP ladder at seven saturation powers
(0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8 µT) and a QUEST ladder at six saturation times
(0.4, 1, 2, 3, 4, 5 s, at 1.0 µT). 101 offsets from −5 to +5 ppm in both.

## Files

**`phantom_4vials.npz`** (21 kB) — the measurement, averaged over each tube's region, plus every
per-vial constant the fit holds fixed:

| | |
|---|---|
| `Z_quesp` (4, 7, 101), `Z_quest` (4, 6, 101) | B0-corrected Z-spectra |
| `ppm`, `b1_levels_uT`, `tsat_s` | what was acquired |
| `R1`, `b0_hz`, `rel_b1` | per tube, measured separately |
| `vial_dw`, `cest_t2_s` | pool offset and pool T2 per tube |
| `water_r2` | the PBS value, used for every tube |
| `f_expected` | pool size implied by the concentration weighed in |
| `m_start` | 0.582 — the readout does not let the sample recover fully between offsets |

These are averages over each tube's region, not the 128×128 maps they came from. The maps are
several hundred times larger and nothing in these tutorials needs them.

**`events/*.npz`** (890 kB) — the sequences, already converted to solver events. Pulseq files are
read once and cached because reading one costs several GB of memory and over a minute; a
conversion is only valid at the `max_pulse_samples` it was built with, which is recorded in each
file. They are the simulation variant of the played sequences: the same saturation module with a
single-sample readout instead of the imaging one, since a forward model needs one point per
offset and not 128 phase-encode lines.

Each file carries one offset more than the Z-spectra: the sequences acquire an unsaturated
reference at −300 ppm first. `_phantom.load_events` drops it.

**`seqs/*.seq`** (11 MB, about 1 MB in the repository once compressed) — the Pulseq files those
events were converted from. The tutorials do not read them; they are here so the sequences are
inspectable and reconvertible rather than something you have to take on trust. Their
`[DEFINITIONS]` block states the B1 levels, every offset, the saturation timing and the gap
spoiling, so each file documents the acquisition it belongs to.

Reconverting one needs pypulseq and costs several GB and about a minute:

```python
import pypulseq as pp
from bmc_jax.jax_solver.seq_adapter import convert_sequence_batch

seq = pp.Sequence()
seq.read("data/seqs/quesp_sim_b1_1uT.seq")
events, n_offsets = convert_sequence_batch(seq, max_pulse_samples=200)
```

`max_pulse_samples` must be the 200 the cache was built with. A different value gives a
different forward model, not an error.

## What is not here

The scanner raw data, the reconstruction, the B0 and B1 maps, the T1 and T2 mapping, and the
per-tube pool-T2 profiling that produced `cest_t2_s`. Those are upstream of these tutorials.
What ships is the input to the fit and nothing else.
