# Shipped phantom measurement

Tutorials 08 to 11 fit real scanner data. This is that data, the sequences that produced it, and
the rest of the scan.

| | what | read by the tutorials |
|---|---|---|
| `phantom_7tubes.npz` | the Z-spectra averaged over each tube, and every per-tube constant | yes |
| `events/` | the simulation sequences, converted to solver events | yes |
| `seqs/` | those simulation sequences as Pulseq files | no |
| [`scan_2026-09-27/`](scan_2026-09-27/README.md) | the sequences as played, the reconstructed images of every voxel, the tube regions, every B0, B1, T1, T2 and M_start map | no |

Only the scanner raw data is not here (Siemens TWIX, 32 files, 6.1 GB): too large for this repository.

## The scan

A seven-tube phantom measured at 2.8936 T on 2026-09-27. One tube is plain PBS, with no CEST pool;
the other six form three pairs, so that a ratio of fitted pool sizes can be checked without knowing
any absolute concentration.

| tube | name | contents | pH | pool offset | `f_nominal` |
|---|---|---|---|---|---|
| 0 | `PBS` | phosphate-buffered saline | 7.76 | none | 0 |
| 1 | `GAG100_pH6.31` | chondroitin sulfate 100 mg/mL | 6.31 | 1.0 ppm, hydroxyl | 5.37 × 10⁻³ |
| 2 | `GAG50_pH6.20` | chondroitin sulfate 50 mg/mL | 6.20 | 1.0 ppm, hydroxyl | 2.69 × 10⁻³ |
| 3 | `creatine_pH7.51` | creatine 90 mM | 7.51 | 1.9 ppm, guanidinium | 3.24 × 10⁻³ |
| 4 | `creatine_pH6.31` | creatine 90 mM | 6.31 | 1.9 ppm, guanidinium | 3.24 × 10⁻³ |
| 5 | `carnosine_pH6.59` | carnosine 50 mM | 6.59 | 3.2 ppm, amide | 4.5 × 10⁻⁴ |
| 6 | `carnosine_pH7.44` | carnosine 50 mM | 7.44 | 3.2 ppm, amide | 4.5 × 10⁻⁴ |

The pairs: the two GAG tubes come from one stock at 100 and 50 mg/mL, so their pool sizes must have
the ratio 2. The two creatine tubes are one solution at two pH, and so are the two carnosine tubes,
so each of those ratios must be 1. A calibration error shared by both tubes of a pair cancels in the
ratio; that is what makes these checks worth more than any absolute number.

`f_nominal` is the concentration on the label times the labile protons per molecule (creatine 4,
GAG 3 per disaccharide unit, carnosine 1) over 111 M, the proton concentration of water. GAG is chondroitin
sulfate sodium salt, converted at 503 g/mol per disaccharide unit. The masses were not recorded, so it is
nominal: good for ratios, not a weighed truth.

Each pH was measured once, the day before the scan. The phantom was at 21.9 to 22.0 °C before the
first scan and 22.4 to 22.5 °C after the last, 4 h 46 min later.

The tubes were measured twice: a QUESP ladder at eight saturation powers
(0.15, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8 µT) and a QUEST ladder at six saturation times
(0.4, 1, 2, 3, 4, 5 s, at 1.0 µT). 101 offsets from −5 to +5 ppm in both.

## Files

**`phantom_7tubes.npz`** (38 kB): the measurement, averaged over each tube, plus every per-tube
constant a fit can hold fixed. One entry per tube along the first axis, in the order of the table above.

| | |
|---|---|
| `Z_quesp` (7, 8, 101), `Z_quest` (7, 6, 101) | Z-spectra, B0-corrected (below) |
| `ppm`, `b1_levels_uT`, `tsat_s` | what was acquired; powers in ascending order |
| `vial_names`, `vial_labels`, `pH` | which tube is which |
| `vial_dw` | pool offset in ppm; NaN for PBS (no pool) |
| `f_nominal` | pool size implied by the label (above) |
| `R1` | water R1 (1/s): 1 / the mean of two T1 maps, one before QUESP and one after QUEST |
| `R1_start`, `R1_end` | the same from each map alone |
| `rel_b1` | relative B1 from a WASABI map |
| `b0_hz` | 0 for every tube: the B0 correction is already applied to the spectra |
| `water_r2` | water R2 (1/s) of the PBS tube from a CPMG train (40 ms echo spacing), used for every tube |
| `m_start` | 0.557: the magnetisation at the start of saturation, as a fraction of equilibrium (below) |
| `noise_quesp`, `noise_quest` | noise of a tube-mean spectrum, per ladder (below) |
| `n_voxels` | voxels averaged per tube |
| `b0_T` | 2.8936 |

**How the spectra were made.** Each voxel's spectrum was shifted by the voxel's B0 from a WASABI map,
and each rung by the slow field drift during the session, measured from the water line. The spectra
are then averaged over the core of each tube, 23 to 28 voxels away from the tube wall.

**`m_start`.** The readout does not let the sample recover fully between offsets, so saturation does
not begin at equilibrium. 0.557 was fitted to the PBS tube's QUEST spectra alone: PBS has no pool, so
no exchange rate or pool size can trade with it. Tutorials 08 and 09 hold it; 10 and 11 fit it.

**`noise_quesp`, `noise_quest`.** Each voxel's scatter over the far-wing offsets (|ppm| ≥ 4.5), the
median over the tube, divided by the square root of `n_voxels`. A fit whose RMS misfit equals it fits
to the noise.

**Not measured: the pool T2.** No pool T2 was profiled for this phantom; the tutorials fit it per tube
from 10 ms.

These are averages over each tube. The 128×128 images and maps they came from, and how the spectra
were corrected, are in [`scan_2026-09-27/`](scan_2026-09-27/README.md).

**`events/*.npz`** (960 kB, 8 QUESP + 6 QUEST rungs): the sequences, already converted to solver events.
Pulseq files are read once and cached because reading one costs several GB of memory and over a
minute; a conversion is only valid at the `max_pulse_samples` it was built with, which is recorded in
each file. They are the simulation variant of the played sequences: the same saturation module with a
single-sample readout instead of the imaging one, since a forward model needs one point per offset and
not 128 phase-encode lines.

Each file carries one offset more than the Z-spectra: the sequences acquire an unsaturated
reference at −300 ppm first. `_phantom.load_events` drops it.

**`seqs/*.seq`** (12 MB, about 1 MB in the repository once compressed): the Pulseq files those
events were converted from. The tutorials do not read them; they are here so the sequences are
inspectable and reconvertible rather than something you have to take on trust. Their
`[DEFINITIONS]` block states the B1 levels, every offset, the saturation timing and the gap
spoiling, so each file documents the acquisition it belongs to.

Reconverting one needs pypulseq and costs several GB and about a minute:

```python
import pypulseq as pp
from bmc_jax.sequences import events_from_seq_batch

seq = pp.Sequence()
seq.read("data/seqs/quesp_sim_b1_1uT.seq")
events, n_offsets = events_from_seq_batch(seq, max_pulse_samples=200)
```

`max_pulse_samples` must be the 200 the cache was built with. A different value gives a
different forward model, not an error.

## What is not here

The scanner raw data (Siemens TWIX, 6.1 GB), and the analysis code that turned it into these files.
