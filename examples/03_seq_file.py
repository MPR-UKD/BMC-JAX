"""Tutorial 03: Simulating a Pulseq sequence file

Tutorial 02 built the saturation sequence in code. On the scanner, the sequence is defined by a
Pulseq .seq file. Simulating the same file that the scanner plays ensures that the simulation uses
the same pulses, timings and frequency offsets as the measurement.

This tutorial
  1. writes a simple saturation sequence with pypulseq: one block pulse per offset, each followed
     by an ADC event;
  2. reads the file back and converts it into simulation events with events_from_seq_batch();
  3. simulates the Z-spectrum and compares it with the same sequence built in code.

The ADC event marks the end of each offset. The converter splits the sequence at every ADC, so that
each offset becomes one row of events, and all rows are simulated in parallel. Shaped pulses are
divided into at most max_pulse_samples steps of constant amplitude.

Requirements: pypulseq, installed with  pip install "bmc-jax[pulseq]".

Run:  python 03_seq_file.py
"""
import numpy as np
import pypulseq as pp

from bmc_jax.io import example_config
from bmc_jax.simulation import to_sim_params, initial_state, simulate_spectrum
from bmc_jax.sequences import cw_saturation
from bmc_jax.sequences.pulseq import events_from_seq_batch

print(__doc__)

params = example_config("amide")
jax_params = to_sim_params(params)
m0 = initial_state(jax_params)

PPM = np.linspace(-6.0, 6.0, 25)
B1_UT = 1.0
TSAT_S = 2.0
GAMMA_HZ_PER_T = 42.5764e6
hz_per_ppm = GAMMA_HZ_PER_T * params.system.b0 / 1e6

# ---------------------------------------------------------------------------------------------
# 1. Write a .seq file
# ---------------------------------------------------------------------------------------------
# One saturation block per offset, each followed by an ADC event. The ADC marks the end of an
# offset: events_from_seq_batch splits the sequence at every ADC event, which gives one row per
# offset, and jax.vmap runs the rows in parallel.

system = pp.Opts(rf_ringdown_time=0, rf_dead_time=0, rf_raster_time=1e-6)
seq = pp.Sequence(system=system)

# A block pulse of amplitude B1 for TSAT_S. pypulseq wants a flip angle, so convert:
# flip [rad] = gamma[rad/s/T] * B1[T] * duration[s]
gamma_rad = 2 * np.pi * GAMMA_HZ_PER_T
flip = gamma_rad * (B1_UT * 1e-6) * TSAT_S

for offset_ppm in PPM:
    sat = pp.make_block_pulse(flip_angle=flip, duration=TSAT_S,
                              freq_offset=offset_ppm * hz_per_ppm, system=system)
    seq.add_block(sat)
    seq.add_block(pp.make_adc(num_samples=1, duration=1e-5, system=system))

seq.write("03_saturation.seq")
print(f"wrote 03_saturation.seq: {len(PPM)} offsets, B1 {B1_UT} uT, Tsat {TSAT_S} s")

# ---------------------------------------------------------------------------------------------
# 2. Read the file back and convert it
# ---------------------------------------------------------------------------------------------
# max_pulse_samples sets how finely shaped pulses are discretised. Every block is padded to this
# length, so that all blocks have the same shape and the whole sequence compiles into one loop.
# A stored conversion is valid only for the value it was built with. Mixing values gives a wrong
# simulation and no error, so store the value together with any converted sequence.

seq_in = pp.Sequence(system=system)
seq_in.read("03_saturation.seq")
events, n_offsets = events_from_seq_batch(seq_in, max_pulse_samples=params.options.max_pulse_samples)
print(f"read back: {n_offsets} offsets x {events.dt.shape[1]} events per offset")

z_from_seq = np.asarray(simulate_spectrum(jax_params, events, m0,
                                          return_full_trajectory=False)[:, params.mz_loc])

# ---------------------------------------------------------------------------------------------
# 3. Compare with the sequence built in code
# ---------------------------------------------------------------------------------------------
# The same experiment described in two ways. The results are not identical: the .seq path divides
# the pulse into samples on the RF raster, while the code path uses one block of constant
# amplitude. They should agree closely. A large difference means that the two do not describe the
# same experiment.

events_code = cw_saturation(PPM, b1_uT=B1_UT, tsat_s=TSAT_S, b0_T=params.system.b0)
z_from_code = np.asarray(simulate_spectrum(jax_params, events_code, m0,
                                           return_full_trajectory=False)[:, params.mz_loc])

diff = np.abs(z_from_seq - z_from_code)
print(f"\n{'ppm':>7} {'from .seq':>10} {'in code':>10} {'diff':>9}")
for i in range(0, len(PPM), 3):
    print(f"{PPM[i]:>+7.2f} {z_from_seq[i]:>10.5f} {z_from_code[i]:>10.5f} {diff[i]:>9.2e}")
print(f"\nmax difference {diff.max():.2e}, caused by the pulse discretisation of the .seq path")

print("\nNext: 04_fit_one.py, fitting k and f from a single saturation level.")
