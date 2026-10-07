"""03 -- Driving the simulation from a real Pulseq .seq file.

Tutorial 02 built the sequence in code. That is ideal for understanding and for tests, but a
scanner runs a .seq file, and simulating exactly the sequence you will execute is the point of
a digital twin. This writes one with pypulseq, reads it back, and simulates it.

Needs pypulseq:  pip install "bmc-jax[pulseq]"
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
# 1. Write a .seq
# ---------------------------------------------------------------------------------------------
# One saturation block per offset, each followed by an ADC. The ADC is what marks the end of an
# offset: events_from_seq_batch splits the sequence on ADC events, giving one row per offset
# that vmap can run in parallel.

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
print(f"wrote 03_saturation.seq -- {len(PPM)} offsets, B1 {B1_UT} uT, Tsat {TSAT_S} s")

# ---------------------------------------------------------------------------------------------
# 2. Read it back and convert
# ---------------------------------------------------------------------------------------------
# max_pulse_samples decides how finely shaped pulses are discretised. Every block is padded to
# that length so each has the same static shape and the whole sequence compiles to one loop.
# A cached conversion is only valid at the value it was built with -- mixing them gives a wrong
# simulation rather than an error, so it is worth recording alongside any cache.

seq_in = pp.Sequence(system=system)
seq_in.read("03_saturation.seq")
events, n_offsets = events_from_seq_batch(seq_in, max_pulse_samples=params.options.max_pulse_samples)
print(f"read back: {n_offsets} offsets x {events.dt.shape[1]} events per offset")

z_from_seq = np.asarray(simulate_spectrum(jax_params, events, m0,
                                          return_full_trajectory=False)[:, params.mz_loc])

# ---------------------------------------------------------------------------------------------
# 3. Compare against the in-code sequence
# ---------------------------------------------------------------------------------------------
# The same physics described two ways. They will not be bit-identical -- the .seq path
# discretises the pulse into samples on a raster, the in-code path is one analytic block -- but
# they should agree closely. A large disagreement means the two are not the experiment you think.

events_code = cw_saturation(PPM, b1_uT=B1_UT, tsat_s=TSAT_S, b0_T=params.system.b0)
z_from_code = np.asarray(simulate_spectrum(jax_params, events_code, m0,
                                           return_full_trajectory=False)[:, params.mz_loc])

diff = np.abs(z_from_seq - z_from_code)
print(f"\n{'ppm':>7} {'from .seq':>10} {'in code':>10} {'diff':>9}")
for i in range(0, len(PPM), 3):
    print(f"{PPM[i]:>+7.2f} {z_from_seq[i]:>10.5f} {z_from_code[i]:>10.5f} {diff[i]:>9.2e}")
print(f"\nmax difference {diff.max():.2e}  --  pulse discretisation, not a disagreement")

print("\nNext: 04_fit_one.py -- recover k and f from a spectrum, and see why one is not enough.")
