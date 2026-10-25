"""Tutorial 01: Defining the model

A Bloch-McConnell simulation describes the magnetisation of several proton pools that exchange
protons with water. Before anything can be simulated, the model has to be defined: which pools
exist, their size relative to water, their relaxation times, their chemical shift and their
exchange rate, together with the field strength and the numerical options.

This tutorial shows the two ways to define a model in BMC-JAX:
  1. in code, with the classes WaterPool, CESTPool, MTPool, System, Options and Parameters;
  2. from a Pulseq-CEST YAML configuration file, with from_yaml() or example_config().
Both give the same Parameters object. The tutorial then shows that the objects cannot be changed
after they are created, and that every value is checked when the object is created.

Quantities used
  f       pool size relative to water (proton fraction)
  k       exchange rate from the pool to water, in 1/s (printed as Hz)
  dw      chemical shift of the pool relative to water, in ppm
  T1, T2  relaxation times in s; alternatively the rates R1 = 1/T1 and R2 = 1/T2 in 1/s

Requirements: the core package only.

Run:  python 01_pools.py
"""
from bmc_jax.io import (CESTPool, MTPool, Options, Parameters, System, WaterPool,
                        example_config, example_config_names, from_yaml)

print(__doc__)

# ---------------------------------------------------------------------------------------------
# 1. In code
# ---------------------------------------------------------------------------------------------
# Relaxation can be given as times (t1, t2 in seconds) or as rates (r1, r2 in 1/s). Giving both for
# the same quantity is rejected: if the two disagree, one of them is wrong, and silently preferring
# one would hide the error.

water = WaterPool(f=1.0, t1=1.3, t2=0.05)
amide = CESTPool(f=0.0009, t1=1.0, t2=0.1, k=50, dw=3.5)   # 3.5 ppm downfield of water
noe = CESTPool(f=0.0045, t1=1.3, t2=0.005, k=20, dw=-3.0)  # upfield of water: a negative shift
mt = MTPool(f=0.05, t1=1.0, t2=1e-5, k=30, dw=-2.5, lineshape="SuperLorentzian")

params = Parameters(water_pool=water, cest_pools=[amide, noe], mt_pool=mt,
                    system=System(b0=3.0), options=Options(max_pulse_samples=200))

print("built in code")
print(f"  water      T1 {water.T1:.2f} s   T2 {water.T2:.3f} s   r1 {water.r1:.3f} Hz")
print(f"  amide      f {amide.f:.4%}   k {amide.k:.0f} Hz   dw {amide.dw:+.1f} ppm")
print(f"  NOE        f {noe.f:.4%}   k {noe.k:.0f} Hz   dw {noe.dw:+.1f} ppm")
print(f"  MT         f {mt.f:.2%}     lineshape {mt.lineshape}")
print(f"  scanner    B0 {params.system.b0} T   rel_B1 {params.system.rel_b1}")
print(f"  pools      {params.num_cest_pools} CEST + MT     water Mz lives at index {params.mz_loc}")

# ---------------------------------------------------------------------------------------------
# 2. From a Pulseq-CEST YAML
# ---------------------------------------------------------------------------------------------
# The same description in the Pulseq-CEST configuration format. Four configurations ship with the
# package, so that all tutorials run without files of your own.

print(f"\nconfigs shipped with the package: {example_config_names()}")
for name in example_config_names():
    p = example_config(name)
    mt_note = p.mt_pool.lineshape if p.mt_pool else "-"
    print(f"  {name:20s} {p.num_cest_pools} CEST   MT {mt_note:16s} B0 {p.system.b0} T")

# from_yaml() reads your own file the same way:
#     params = from_yaml("my_model.yaml")

# ---------------------------------------------------------------------------------------------
# 3. Immutability and validation
# ---------------------------------------------------------------------------------------------
# Every value is checked once, when the object is created. Because the object cannot be changed
# afterwards, this single check is sufficient: no later assignment can bypass it. To change a
# value, create a modified copy with dataclasses.replace().

from dataclasses import FrozenInstanceError, replace

try:
    amide.k = 500.0
except FrozenInstanceError:
    print("\nassigning to a pool raises an error; create a modified copy instead:")
    print(f"  replace(amide, k=500).k = {replace(amide, k=500).k}")

for bad, why in [(dict(f=1.5), "a pool fraction above 1"),
                 (dict(k=-10), "a negative exchange rate"),
                 (dict(t1=1.0, r1=1.0), "both a time and a rate for the same thing")]:
    try:
        CESTPool(**{**dict(f=0.001, t1=1.0, t2=0.1, k=50, dw=3.5), **bad})
    except (ValueError, TypeError) as exc:
        print(f"  rejected: {why}\n            {exc}")

print("\nNext: 02_simulate.py, simulating a Z-spectrum from this model.")
