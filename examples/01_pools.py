"""01 -- Describing what you are simulating: pools, scanner, options.

A Bloch-McConnell simulation needs to know which proton pools exist, how big they are, where
they resonate and how fast they exchange. There are two ways to say that, and they produce the
same object.

Run:  python 01_pools.py
"""
from bmc_jax.io import (CESTPool, MTPool, Options, Parameters, System, WaterPool,
                        example_config, example_config_names, from_yaml)

print(__doc__)

# ---------------------------------------------------------------------------------------------
# 1. In code
# ---------------------------------------------------------------------------------------------
# Relaxation can be given as TIMES (t1, t2 in seconds) or as RATES (r1, r2 in Hz), whichever you
# have. Never both: if the two disagree, one of them is a mistake, and quietly preferring one
# would hide it.

water = WaterPool(f=1.0, t1=1.3, t2=0.05)
amide = CESTPool(f=0.0009, t1=1.0, t2=0.1, k=50, dw=3.5)   # 3.5 ppm downfield of water
noe = CESTPool(f=0.0045, t1=1.3, t2=0.005, k=20, dw=-3.0)  # upfield -- a negative shift
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
# The same description, in the format the community uses. Four configs ship with the package so
# you can run everything here without files of your own.

print(f"\nconfigs shipped with the package: {example_config_names()}")
for name in example_config_names():
    p = example_config(name)
    mt_note = p.mt_pool.lineshape if p.mt_pool else "-"
    print(f"  {name:20s} {p.num_cest_pools} CEST   MT {mt_note:16s} B0 {p.system.b0} T")

# from_yaml() reads your own file the same way:
#     params = from_yaml("my_model.yaml")

# ---------------------------------------------------------------------------------------------
# 3. They are immutable, and every value is checked
# ---------------------------------------------------------------------------------------------
# Validation runs once, at construction. Freezing is what makes once enough: an object that
# cannot be modified cannot drift into a state nothing ever checked.

from dataclasses import FrozenInstanceError, replace

try:
    amide.k = 500.0
except FrozenInstanceError:
    print("\nassigning to a pool raises -- build a new one instead:")
    print(f"  replace(amide, k=500).k = {replace(amide, k=500).k}")

for bad, why in [(dict(f=1.5), "a pool fraction above 1"),
                 (dict(k=-10), "a negative exchange rate"),
                 (dict(t1=1.0, r1=1.0), "both a time and a rate for the same thing")]:
    try:
        CESTPool(**{**dict(f=0.001, t1=1.0, t2=0.1, k=50, dw=3.5), **bad})
    except (ValueError, TypeError) as exc:
        print(f"  rejected: {why}\n            {exc}")

print("\nNext: 02_simulate.py -- turn this into a Z-spectrum.")
