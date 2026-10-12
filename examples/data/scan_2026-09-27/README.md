# The 2026-09-27 phantom scan: everything except the raw data

The tutorials read only `../phantom_7tubes.npz` and `../events/`. This folder holds the rest of the
scan they were made from: the sequences as played, the reconstructed images and spectra, and every
map. The tubes, their contents and pH are described in [`../README.md`](../README.md).

Scanner: 2.8936 T. Session 14:23:38 to 19:09:58 (4 h 46 min, scanner clock). Matrix 128 × 128, one slice.

| | what | size |
|---|---|---|
| `seqs_played/` | the 25 Pulseq files built for the session (`.seq`), as played on the scanner | 49 MB |
| `reko/` | reconstructed Z-spectra, every voxel, both ladders; the tube regions; the tube-mean spectra | 75 MB |
| `maps.npz` | B0, B1, T1, T2 and M_start maps, the set the fits used (one array per map) | 0.5 MB |
| `maps_repeats.npz` | the maps measured a second time in the session (drift check) | 0.4 MB |

**Not here: the scanner raw data** (Siemens TWIX `.dat`, 32 files, 6.1 GB). It is too large for this
repository.

## What was acquired

In acquisition order. `MID` is the scanner's measurement number; the raw files are named after it.

| MID | time | sequence (`seqs_played/`) | what | used for |
|---:|---|---|---|---|
| 24 | 14:23 | `b0map_gre.seq` | B0, gradient echo | `maps.npz`: `b0map_gre` |
| 25 | 14:25 | `b1map_dam.seq` | B1, double angle | `maps.npz`: `b1dam_relb1` |
| 26 | 14:26 | `mstart_recovery.seq` | M_start, recovery series | `maps.npz`: `mstart_*` |
| 27 | 14:28 | `wassr.seq` (0.10 µT) | WASSR | stopped by the scanner at the first pulse: no data |
| 28 | 14:30 | `wasabi.seq` | WASABI #1 | `maps_repeats.npz`: `wasabi_*` |
| 29 | 14:32 | `wassr.seq` (0.10 µT) | WASSR, retry | stopped again: no data |
| 30 | 14:35 | an older WASSR build at 0.30 µT (not in `seqs_played/`) | WASSR | `maps.npz`: `wassr_b0` |
| 31 | 14:38 | `wasabi.seq` | WASABI #2 | `maps.npz`: `wasabi_*` (the B0 and B1 the fits use) |
| 32 | 14:41 | `t1_map_ir.seq` | T1, inversion recovery | `maps.npz`: `t1_map` |
| 33–40 | 14:43–16:44 | `quesp_scanner_b1_*.seq` | QUESP: 0.3, 1.8, 0.8, 1.3, 0.5, 1.0, 0.15, 1.5 µT (in that order) | `reko/quesp_Z.npz` |
| 41–46 | 16:44–18:20 | `quest_scanner_tsat_*.seq` | QUEST at 1.0 µT: 1, 5, 3, 2, 0.4, 4 s (in that order) | `reko/quest_Z.npz` |
| 47 | 18:20 | `t2_disp_esp_20ms.seq` | T2, CPMG, 20 ms echo spacing | `maps.npz`: `t2_map_disp20` |
| 48 | 18:34 | `t2_disp_esp_40ms.seq` | T2, CPMG, 40 ms | `maps.npz`: `t2_map_disp40` (water R2 of the fits) |
| 49 | 18:44 | `t2_disp_esp_80ms.seq` | T2, CPMG, 80 ms | `maps.npz`: `t2_map_disp80` |
| 50 | 18:54 | `t1_map_ir.seq` | T1, repeat | `maps_repeats.npz`: `t1_map` |
| 51 | 18:56 | `b1map_dam.seq` | B1, repeat | `maps_repeats.npz`: `b1dam_relb1` |
| 52 | 18:57 | `mstart_recovery.seq` | M_start, repeat | `maps_repeats.npz`: `mstart_*` |
| 53 | 18:59 | `b0map_gre.seq` | B0, repeat | `maps_repeats.npz`: `b0map_gre` |
| 54 | 18:59 | `t2_disp_esp_150ms.seq` | T2, CPMG, 150 ms | `maps.npz`: `t2_map_disp150` |
| 55 | 19:09 | `t2_disp_esp_10ms.seq` | T2, CPMG, 10 ms | stopped after 1 s: no data |

Every CEST rung acquires an unsaturated M0 image (offset −300 ppm) and then 101 offsets from −5 to
+5 ppm.

## `reko/`

| file | arrays | what |
|---|---|---|
| `quesp_Z.npz` | `Z` (8, 101, 128, 128), `M0` (8, 128, 128), `ppm`, `b1_levels_uT`, `rep_order` | QUESP, every voxel. Each rung divided by its own M0 image. **Not B0-corrected.** Rungs in the order of `b1_levels_uT` (0.3 … 1.8, then 0.15) |
| `quest_Z.npz` | `Z` (6, 101, 128, 128), `M0` (6, 128, 128), `ppm`, `tsat_s`, `rep_order` | QUEST, the same |
| `quesp_id_vial_spectra.npz` | `labels` (128, 128), `vial_names`, `centroids`, `angles`, `spec`, `asym` | the tube regions: `labels` is 0 outside the tubes and tube index + 1 inside (the core of each tube, away from the wall); the uncorrected tube means and their MTRasym |
| `quesp_b0corr_vial_spectra.npz` | `spec`, `spec_raw`, `spec_roi_corrected`, `offset_hz`, `drift_coef`, `drift_t0`, `b0_map`, `labels` | QUESP tube means: `spec` B0-corrected voxel by voxel before averaging (what the fits use), `spec_raw` uncorrected, `spec_roi_corrected` corrected once after averaging (for comparison) |
| `quest_b0corr_vial_spectra.npz` | `spec`, `spec_raw`, `offset_hz`, `pbs_after_hz`, `drift_coef`, `drift_t0`, `b0_map`, `labels` | QUEST tube means, the same correction |

**The B0 correction.** Each voxel's spectrum is shifted by its value in the `wasabi_b0` map (`maps.npz`). On top
of that, each rung gets one common offset (`offset_hz`), because the field drifted by about −19.5 Hz
over the session with the transmitter frequency held fixed. The drift is a quadratic in clock time
(`drift_coef`, in clock hours relative to `drift_t0`) through the PBS tube's water line at all 14 CEST rungs; PBS is
the reference because it has no pool to pull the line. The drift within one 15-minute rung (about
1.4 Hz) is not removed.

## `maps.npz` and `maps_repeats.npz`

One array per map, all 128 × 128, NaN where a map has no signal:
`np.load("maps.npz")["wasabi_b0"]`.

| array | unit | what |
|---|---|---|
| `wasabi_b0` | Hz | B0 from WASABI: the map used for the correction |
| `wasabi_relb1` | relative | B1 from WASABI: the `rel_b1` of the fits (tube medians) |
| `b0map_gre` | Hz | B0 from the gradient-echo map |
| `wassr_b0` | Hz | B0 from WASSR (0.30 µT) |
| `b1dam_relb1` | relative | B1 from the double-angle method |
| `t1_map` | s | T1, inversion recovery. The fits use 1 / the mean of the two T1 maps' tube medians |
| `t2_map_disp20`, `_disp40`, `_disp80`, `_disp150` | s | T2 from CPMG trains at 20, 40, 80 and 150 ms echo spacing |
| `mstart_m` | fraction | M_start from the recovery series, **not usable**: the single-exponential model behind it fails its own check (its T1 disagrees with the inversion-recovery T1 by up to about 20 %), and it gives 0.37–0.41 where the CEST data want 0.53–0.56 |
| `mstart_t1` | s | the T1 of that same recovery fit, the number the check compares |

`maps_repeats.npz` holds the same maps from MID 28 (WASABI #1, early in the session) and MID 50–53 (at
the end), for checking drift. Same array names, same units.
