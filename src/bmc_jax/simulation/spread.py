"""spread.py -- a voxel whose water does not sit at one frequency: the B0 spread inside it.

`simulate_spectrum` follows ONE isochromat: every spin of the sample sees the same field. A real voxel
(or a tube, or a phantom compartment) holds spins at slightly different fields, a few Hz apart. Under a
pulsed saturation train a single isochromat rings: the Z-spectrum near water and under strong
saturation gains ripples and edges that the measurement, an average over the voxel, does not show.

`B0Spread` describes the distribution of the field offset inside the sample: its shape, its width
`sigma_hz`, and how many isochromats (`nodes`) represent it. `simulate_spectrum_spread` simulates the
sample at every node -- the whole sample shifted by sigma * u_j -- and returns the weighted mean of the
magnetisation, which is what the scanner measures.

  shape "gauss"    normal distribution, sigma = its SD. Nodes and weights are Gauss-Hermite
                   (probabilists'), exact for a polynomial of degree 2 * nodes - 1 in the offset.
  shape "lorentz"  Lorentzian, sigma = its half width at half maximum. Its tails never end, so it is cut at
                   +-`cut` sigma; with u = tan(t), the Lorentzian is uniform in t, and Gauss-Legendre nodes
                   over t in +-atan(cut) are used. Weights renormalised to 1 after the cut.

The width is a property of the sample, not of the sequence, so it can be fitted like any other: in
`fit_joint` the field "b0_spread" (Hz) is a free parameter, usually one value shared by every sample.

A caution learned on real data: the fitted sigma depends on the node count (on one phantom 3.1 / 4.4 /
5.8 Hz at 3 / 5 / 7 Gauss-Hermite nodes, with the same fit quality at 5 and 7). Treat it as an
effective width for that node count, not as a measurement of the field distribution, and keep the
count fixed when comparing fits.
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from .simulate import simulate_spectrum

SHAPES = ("gauss", "lorentz")


@dataclass(frozen=True)
class B0Spread:
    """The distribution of the field offset inside a sample.

    Parameters
    ----------
    sigma_hz
        Width in Hz: the SD for "gauss", the half width at half maximum for "lorentz". The value used
        where `fit_joint` does not fit it.
    shape
        "gauss" or "lorentz".
    nodes
        Isochromats per sample. The cost of a simulation is `nodes` times one isochromat's.
    cut
        "lorentz" only: the distribution is cut at +-cut widths.
    """
    sigma_hz: float = 0.0
    shape: str = "gauss"
    nodes: int = 5
    cut: float = 20.0

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"B0Spread shape {self.shape!r}; use one of {SHAPES}")
        if int(self.nodes) < 1:
            raise ValueError(f"B0Spread nodes {self.nodes}: at least 1")
        if self.shape == "lorentz" and not self.cut > 0:
            raise ValueError(f"B0Spread cut {self.cut}: must be positive")

    def nodes_weights(self):
        """(u, w): offsets in units of sigma, and weights summing to 1 (numpy, float64)."""
        if self.shape == "gauss":
            u, w = np.polynomial.hermite_e.hermegauss(int(self.nodes))
        else:
            t, w = np.polynomial.legendre.leggauss(int(self.nodes))
            u = np.tan(t * np.arctan(self.cut))
        return u, w / w.sum()


def simulate_spectrum_spread(params, events, M_init, sigma_hz, spread):
    """The final state of every offset, averaged over the isochromats of `spread` with width `sigma_hz`.

    Parameters
    ----------
    params, events, M_init
        As for `simulate_spectrum` (one sample).
    sigma_hz
        The width in Hz (a scalar; may be traced, so it can be fitted).
    spread
        A `B0Spread`: shape and node count (its own `sigma_hz` is not used here).

    Returns
    -------
    ndarray
        (offsets, state) like `simulate_spectrum(..., return_full_trajectory=False)`: the weighted mean
        over the nodes. With sigma 0 every node is the plain simulation, and so is the mean.
    """
    dt = params.water_r1.dtype
    u, w = spread.nodes_weights()
    uw = jnp.asarray(np.stack([u, w], axis=1), dt)
    hz_per_ppm = params.gamma * params.b0 / (2 * np.pi)       # gamma in rad/s/uT (matrix.py): 127.7 Hz/ppm at 3 T
    shape = jax.eval_shape(lambda: simulate_spectrum(params, events, M_init, return_full_trajectory=False))

    def node(acc, a):
        q = params._replace(b0_inhom=params.b0_inhom + sigma_hz * a[0] / hz_per_ppm)
        return acc + a[1] * simulate_spectrum(q, events, M_init, return_full_trajectory=False), None

    out, _ = jax.lax.scan(node, jnp.zeros(shape.shape, shape.dtype), uw)
    return out
