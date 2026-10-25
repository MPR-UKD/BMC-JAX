"""Tutorial 02: Simulating a Z-spectrum

A Z-spectrum gives the water signal that remains after saturation, normalised to the unsaturated
signal, as a function of the saturation frequency offset from water (in ppm). It shows a dip at
water (direct water saturation) and a smaller dip at the chemical shift of every exchanging pool
(the CEST effect). Quantitative CEST analysis works on this curve.

This tutorial
  1. loads a model with water, an amide pool (+3.5 ppm), an NOE pool (-3.0 ppm) and an MT pool;
  2. builds a continuous-wave saturation sequence in code with cw_saturation();
  3. simulates the Z-spectrum with simulate_spectrum();
  4. computes the asymmetry MTRasym(dw) = Z(-dw) - Z(+dw), and shows why it does not measure a single
     pool when pools lie on both sides of water;
  5. removes one pool at a time to show what each pool contributes.

The two parameters of the saturation, the amplitude B1 and the saturation time Tsat, are the two
axes of quantitative CEST: varying B1 gives a QUESP experiment (tutorial 05), varying Tsat a QUEST
experiment (tutorial 06).

Requirements: the core package; matplotlib for the optional figure.

Run:  python 02_simulate.py          (add --plot to save a figure, needs matplotlib)
"""
import sys

import numpy as np

from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state, simulate_spectrum
from bmc_jax.sequences import cw_saturation

print(__doc__)

params = example_config("amide_noe_mt")     # water + amide (+3.5) + NOE (-3.0) + MT
jax_params = to_sim_params(params)
m0 = initial_state(jax_params)

ppm = np.linspace(-6.0, 6.0, 121)

# ---------------------------------------------------------------------------------------------
# The sequence
# ---------------------------------------------------------------------------------------------
# cw_saturation builds the events in code: one continuous-wave block per offset, then a readout.
# No .seq file is needed. (Tutorial 03 does the same from a Pulseq file.)
#
# The two parameters of the saturation:
#   b1_uT   saturation amplitude; varying it gives a QUESP experiment
#   tsat_s  saturation time; varying it gives a QUEST experiment

events = cw_saturation(ppm, b1_uT=1.0, tsat_s=2.0, b0_T=params.system.b0)
print(f"sequence: {events.dt.shape[0]} offsets x {events.dt.shape[1]} events")

# params.mz_loc is the index of the water Mz in the solver's state vector. Do not replace it by a
# constant: it depends on the number of pools, and a fixed index would read the wrong pool as soon
# as the model changes.
z = np.asarray(simulate_spectrum(jax_params, events, m0, return_full_trajectory=False)[:, params.mz_loc])

# ---------------------------------------------------------------------------------------------
# Reading the spectrum
# ---------------------------------------------------------------------------------------------
# MTRasym is the classic CEST contrast: the difference between the two sides of the spectrum. It is
# not a measure of a single pool. An NOE pool at -3.0 ppm reduces the asymmetry of an amide pool at
# +3.5 ppm, as the numbers below show.

mtr_asym = z[::-1] - z

print(f"\nfar wing Z(-6 ppm)     {z[0]:.4f}   (nothing saturated, so close to 1)")
print(f"water    Z( 0 ppm)     {z[len(z)//2]:.4f}   (direct saturation)")
for target in (3.5, -3.0):
    i = int(np.argmin(np.abs(ppm - target)))
    print(f"         Z({target:+.1f} ppm)    {z[i]:.4f}   MTRasym {mtr_asym[i]:+.4f}")

print("\nZ-spectrum (offset increases downward, bar = signal remaining)")
for i in range(0, len(ppm), 6):
    bar = "#" * int(round(z[i] * 48))
    print(f"  {ppm[i]:+5.1f} ppm |{bar:<48}| {z[i]:.3f}")

# ---------------------------------------------------------------------------------------------
# One pool at a time
# ---------------------------------------------------------------------------------------------
# Removing one pool and simulating again shows the contribution of that pool to the total
# spectrum.

from dataclasses import replace

print("\ncontribution of each pool at its own offset (Z without it, minus Z with it):")
for name, kept in [("amide", [params.cest_pools[1]]), ("NOE", [params.cest_pools[0]])]:
    stripped = to_sim_params(replace(params, cest_pools=kept))
    z_without = np.asarray(simulate_spectrum(stripped, events, initial_state(stripped),
                                             return_full_trajectory=False)[:, replace(params, cest_pools=kept).mz_loc])
    j = int(np.argmax(np.abs(z_without - z)))
    print(f"  {name:6s} largest effect {abs(z_without - z)[j]:.4f} at {ppm[j]:+.1f} ppm")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    a.plot(ppm, z); a.invert_xaxis(); a.set_ylabel("Z"); a.set_title("Z-spectrum")
    b.plot(ppm, mtr_asym); b.axhline(0, lw=0.5, c="k")
    b.set_xlabel("offset [ppm]"); b.set_ylabel("MTRasym")
    plt.tight_layout(); plt.savefig("02_zspectrum.png", dpi=120)
    print("\nwrote 02_zspectrum.png")

print("\nNext: 03_seq_file.py, the same simulation driven by a Pulseq .seq file.")
