# Tutorials

Eleven scripts introduce BMC-JAX step by step. Each script runs on its own with the installed
package. Tutorials 01 to 07 build all their inputs in code. Tutorials 08 to 11 fit a measured
phantom whose data are included in [`data/`](data/README.md).

```bash
pip install -e ".[pulseq]"     # tutorial 03 needs pypulseq; all others need only the core package
cd examples
python 01_pools.py
```

## Overview

| Tutorial | Content |
|---|---|
| `01_pools.py` | Water, CEST and MT pools, defined in code and read from a Pulseq-CEST YAML configuration. Rates and times, validation, immutability. |
| `02_simulate.py` | From pools to a Z-spectrum. MTRasym, and why it is not a clean measure of a single pool. `--plot` draws a figure (needs matplotlib). |
| `03_seq_file.py` | Writing a Pulseq `.seq` file with pypulseq, reading it back, simulating it, and comparing it with the same sequence built in code. |
| `04_fit_one.py` | Fitting k and f from a single saturation level, and the conditions under which this fails. |
| `05_fit_ladder.py` | The QUESP ladder: `Experiment.stack`, the residual per rung, and `at_bounds`. |
| `06_quest.py` | The QUEST ladder in saturation time. Sequences of different length, and QUESP against QUEST as a cross-check. |
| `07_many_voxels.py` | A slice: per-voxel calibration held fixed, chunked memory use, and the cost of ignoring the maps. |
| `08_quesp.py` | Measured data: six phantom tubes and eight saturation amplitudes. k, f and pool T2 of each tube fitted across the ladder. |
| `09_quest.py` | The same tubes measured with six saturation times. QUESP against QUEST. |
| `10_shared_water.py` | `fit_joint`: a water R2 factor, the start magnetisation and the B0 spread (`B0Spread`) shared by all seven tubes, fitted together with k, f and pool T2 of every pool tube. |
| `11_per_rung.py` | One value per rung of the ladder, shared by the tubes: a B1 factor per saturation amplitude (`scope="block"`), and how to interpret it. |

## One saturation level and the QUESP ladder (04 and 05)

Tutorial 04 shows that a single saturation level can recover k and f from noise-free data. The
separation fails when noise meets a poorly conditioned operating point. The condition is set by the
saturation efficiency, which can be computed before the measurement:

```text
alpha = w1^2 / (w1^2 + k (k + R2s))
```

When `alpha` is close to 1, the pool is already saturated and a change of k has little effect. When
`alpha` is close to 0, the pool is hardly saturated. In both cases the data carry little information
on k. The tutorial fits the same pool at five exchange rates with 0.2 % noise. The errors range from
3 % to 104 %, although every fit converges with an unremarkable residual.

Tutorial 05 shows how a ladder of saturation amplitudes extends the range of exchange rates that can
be measured, and where this range ends: the lowest amplitude sets the limit for slow exchange.

## Measured phantom (08 to 11)

Tutorials 08 to 11 use a seven-tube phantom measured at 3 T on 2026-09-27: one tube of
phosphate-buffered saline and three pairs. The two GAG tubes (100 and 50 mg/mL) should give a ratio
of fitted pool sizes of 2. The creatine and carnosine pairs, each one solution at two pH values,
should give a ratio of 1. A ratio does not depend on the absolute concentration, so these checks are
stronger than any single absolute value.

All inputs are included in [`data/`](data/README.md): the Z-spectra averaged over each tube, the
converted sequences, and the per-tube constants that the fit holds fixed.

Runtimes on one NVIDIA RTX 3070:

| Tutorial | Runtime |
|---|---|
| 08, 09 | about 15 min each (60 fit steps on sequences with several thousand events) |
| 10 | about 2 h (B0 spread with 15 isochromats per tube, 12 joint steps) |
| 11 | about 3.5 h (two joint fits with the B0 spread) |
