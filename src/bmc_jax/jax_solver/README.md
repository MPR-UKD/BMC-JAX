# JAX Solver Architecture

The `jax_solver` module provides the computational backbone for translating and executing Bloch-McConnell equations. By restricting the simulation entirely to JAX primitives, the framework constructs a continuous, fully differentiable computational graph. This structural design evaluates exact analytical derivatives ($\mathbf{J} = \partial \mathbf{Z} / \partial \boldsymbol{\theta}$) via Automatic Differentiation, replacing computationally expensive and numerically unstable finite difference approximations.

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

* `param_adapter.py`: Adapts standard object-oriented representations into strictly statically-padded arrays (`JAXParams`). Implements the construction logic for the $\mathbf{M}_{init}$ initialization maps.
* `seq_adapter.py`: Converts nested `.seq` arrays into flattened `SimEvent` blocks. Events are rigorously padded against shape limits (`max_pulse_samples`) ensuring graph stability for JIT compilation.
* `simulator.py`: Implements exactly the forward time-domain evolution core logic (`simulate_spectrum`), generating sequence-dependent Homogeneous Bloch-McConnell matrices natively invoking the `vmap` + `scan` acceleration architecture.
* `utils.py`: Centralized precision safety ensuring absolute adherence for parameters mapping standard constraints down dynamically to targeted hardware limits (`float32` testing vs `float64` rigor).
* `solver.py`: Executes the pure optimization bindings matching parameters analytically via the evaluated JAX inverse model limits linked iteratively against gradient-based solvers.

<!-- ## Magnetization Vector State

The simulator calculates parameters continuously along standard one-dimensional sequences modeling absolute block-layout dynamics. Assumed against $N_s$ active parameterized sub-pools, the strict non-augmented state evaluates dimensions dynamically:

$$[M_{x,w}, M_{x,s_1} \dots M_{x,s_{N_s}}, M_{y,w}, M_{y,s_1} \dots M_{y,s_{N_s}}, M_{z,w}, M_{z,s_1} \dots M_{z,s_{N_s}}, M_{z,m}]^T$$

*(The engine tracks mathematical offsets dynamically by evaluating $3 \times (N_s + 1) + 1$, targeting semi-solid pools ($Z_m$) safely via dedicated terminus indices).* -->
