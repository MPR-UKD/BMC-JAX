# Tutorials

Seven scripts, in order. Each runs on its own with nothing but the installed package — no data
files, no scanner, no configs of your own.

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
| **05_fit_ladder.py** | The QUESP ladder. `Problem.stack`, residual per rung, `at_bounds`. |
| **06_quest.py** | The other ladder, in saturation time. Ragged sequences, and QUESP vs QUEST as a cross-check. |
| **07_many_voxels.py** | A slice: per-voxel calibration held fixed, chunked memory, and what ignoring the maps costs. |

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
