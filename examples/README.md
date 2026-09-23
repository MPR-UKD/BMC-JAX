# Tutorials

Nine scripts, in order. Each runs on its own with nothing but the installed package. Tutorials
01–07 build everything they need in code; 08 and 09 fit a real phantom measurement, which ships
with them in `data/`.

```bash
pip install -e ".[pulseq]"     # 03 needs pypulseq; the rest need only the core
cd examples
python 01_pools.py
```

| | what it covers |
|---|---|
| **01_pools.py** | Build water, CEST and MT pools in code and from a Pulseq-CEST YAML. Rates or times, validation, immutability. |
| **02_simulate.py** | Pools to Z-spectrum. MTRasym, and why it is not a clean measure of one pool. `--plot` for a figure. |
| **03_seq_file.py** | Write a real `.seq` with pypulseq, read it back, simulate it, and compare against the in-code sequence. |
| **04_fit_one.py** | Fit k and f from one saturation level — and find where that quietly fails. |
| **05_fit_ladder.py** | The QUESP ladder. `Experiment.stack`, residual per rung, `at_bounds`. |
| **06_quest.py** | The other ladder, in saturation time. Ragged sequences, and QUESP vs QUEST as a cross-check. |
| **07_many_voxels.py** | A slice: per-voxel calibration held fixed, chunked memory, and what ignoring the maps costs. |
| **08_quesp.py** | Real data: four phantom tubes, a seven-rung B1 ladder, k and f fitted jointly. Takes a few minutes. |
| **09_quest.py** | The same tubes measured in saturation TIME. Ragged rungs, and QUESP against QUEST as a cross-check. |

## The two worth reading even if you skip the rest

**04** is mostly a demonstration of a failure. On clean data a single saturation level recovers
k perfectly well, so the usual "one level cannot separate k from f" is too strong. What breaks
it is ordinary noise at a badly conditioned operating point, and the condition is computable
before you scan:

```
alpha = w1^2 / (w1^2 + k (k + R2s))
```

Where `alpha` is near 1 the pool is already saturated and raising k changes nothing; where it is
near 0 almost nothing is saturated. Either way the lever is short. The tutorial fits the same
pool at five exchange rates with the same 0.2 % noise and gets errors from 3 % to 104 %, with a
converged fit and a healthy-looking residual every time.

**05** is the fix, and it is honest about its limits: a ladder widens the range of k you can
measure, it does not make it unlimited, and the lowest rung sets the slow end.

## The two that use real data

**08** and **09** fit a four-tube phantom scanned at 2.8936 T. Two tubes hold the same creatine
solution at the same concentration, so their fitted pool sizes must agree — the one check in
these tutorials that no systematic error shared by both tubes can satisfy, and worth more than
any single absolute number.

Everything they need is in [`data/`](data/README.md): the Z-spectra, the converted sequences and
the per-vial constants the fit holds fixed. They are slower than the rest — the sequences are
several thousand events long and there are seven of them — so expect a few minutes, most of it
compilation.
