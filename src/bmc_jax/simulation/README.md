# Simulation: the differentiable Bloch-McConnell engine

The `simulation` package (formerly `jax_solver`, renamed 2026-10-07) provides the computational backbone for translating and executing Bloch-McConnell equations. By restricting the simulation entirely to JAX primitives, the framework constructs a continuous, fully differentiable computational graph. This structural design evaluates exact analytical derivatives ($\mathbf{J} = \partial \mathbf{Z} / \partial \boldsymbol{\theta}$) via Automatic Differentiation, replacing computationally expensive and numerically unstable finite difference approximations.

## Mathematical Formulation

### The Homogeneous Coordinates Formulation

To optimize the solver for XLA compilation, we adopt a **homogeneous coordinate formulation**. By augmenting the state vector with a scalar unity component ($\mathbf{M}_{\text{aug}} = [\mathbf{M}^T, 1]^T$), the affine system translates into a purely linear operator governed by a single matrix exponential:

$$\mathbf{M}_{\text{aug}}(t_{k+1}) = \exp\left( \mathbf{A}_{\text{aug}, k} \Delta t \right) \mathbf{M}_{\text{aug}}(t_k)$$

This reduces all state evolutions to strict sequences of matrix multiplications, entirely removing matrix inversion ($\mathbf{A}^{-1}$) bottlenecks.

## Computational Graph Setup

![Architectural Schematic](DiagrammJAX%20sensitivity.png)

*Architectural Schematic of the Differentiable JAX-based Physics Engine.*

The simulation isolates execution across two fundamental paradigms:

1. **Vectorized Spectral Dimension (`jax.vmap`)**: The generation of a complete $Z$-spectrum is parallelized across all frequency offsets ($\Delta\omega$). Instead of sequentially modeling individual offsets, `vmap` batches them into a single coherent pass utilizing Single Instruction, Multiple Threads (SIMT) parallelism, allowing the entire $Z$-spectrum to be processed in a single GPU pass.
2. **Sequential Temporal Evolution (`jax.lax.scan`)**: The temporal evolution of the magnetization vector within each offset relies on sequential limits. To bypass individual host-to-device dispatch delays inherent to Python loops, `jax.lax.scan` condenses these sequential updates into a unified loop kernel natively compiled within XLA. Magnetization state vectors ($\mathbf{M}_{k,n}$) safely remain in GPU registers throughout the continuous simulation, virtually eliminating memory bandwidth bottlenecks.

## Core Modules

* `params.py`: Adapts the object-oriented pool description into statically padded arrays (`SimParams`, via `to_sim_params`) and builds the initial state (`initial_state`, with `m_start` the magnetisation the readout leaves behind).
* `events.py`: The sequence as the solver sees it (`SimEvent`, `EventType`), padded to fixed shapes for JIT compilation (`pad_events`), cast and repeated.
* `matrix.py`: The homogeneous-coordinate Bloch-McConnell matrix (`bmc_matrix`), with Lorentzian and SuperLorentzian lineshapes.
* `expm.py`: One time step (`propagate`), with the matrix exponential chosen by `bmc_jax.settings.use_expm`: `"library"` (`jax.scipy.linalg.expm`, the default) or `"plain"` (the same float32 algorithm without library calls).
* `simulate.py`: The forward time-domain evolution (`simulate_sequence`, `simulate_spectrum`) with the `vmap` + `scan` architecture above.
* `voxels.py`: The same, batched over voxels or samples with per-voxel parameters.
* `multigpu.py`: Spreads an already-vmapped batch across every local GPU.

Reading a Pulseq `.seq` and building saturation sequences in code live in `bmc_jax.sequences`.

<!-- ## Magnetization Vector State

The simulator calculates parameters continuously along standard one-dimensional sequences modeling absolute block-layout dynamics. Assumed against $N_s$ active parameterized sub-pools, the strict non-augmented state evaluates dimensions dynamically:

$$[M_{x,w}, M_{x,s_1} \dots M_{x,s_{N_s}}, M_{y,w}, M_{y,s_1} \dots M_{y,s_{N_s}}, M_{z,w}, M_{z,s_1} \dots M_{z,s_{N_s}}, M_{z,m}]^T$$

*(The engine tracks mathematical offsets dynamically by evaluating $3 \times (N_s + 1) + 1$, targeting semi-solid pools ($Z_m$) safely via dedicated terminus indices).* -->
