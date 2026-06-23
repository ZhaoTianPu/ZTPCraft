multiloop_fluxonium, oscillator_integrals, decoherence handoff
1) Folder structure
ztpcraft/projects/fluxonium/multiloop_fluxonium/
  __init__.py
  two_loop_fluxoid_model.py
  two_loop_fluxoid_system.py
  two_loop_fluxoid_transition_rates.py
  two_loop_fluxonium_with_arrays_core.py
  two_loop_fluxonium_with_arrays_normal_modes.py
  neb.py
ztpcraft/bosonic/oscillator_integrals/
  __init__.py
  fock_states.py
  oscillators.py
  normal_modes.py
  gaussian_hermite_nd.py
  oscillator_overlap.py
  oscillator_overlap_cache.py
  oscillator_operators.py
  oscillator_system.py
  oscillator_integrals_1d_quadrature.py
  _oscillator_integrals_1d_quadrature.pyx
  _oscillator_integrals_1d_quadrature.c
ztpcraft/decoherence/
  __init__.py
  quantum_noise.py
  fgr.py
2) Key modules and responsibilities
multiloop_fluxonium/two_loop_fluxoid_model.py: Defines fluxoid-sector physics for a 2-loop fluxonium reduced to 1D effective models; builds per-sector scqubits.Fluxonium + local oscillator metadata.

multiloop_fluxonium/two_loop_fluxoid_system.py: High-level system wrapper to get sector energies/eigenvectors and cross-sector overlap matrices.

multiloop_fluxonium/two_loop_fluxoid_transition_rates.py: Operator algebra + vectorized FGR rate machinery over many sector/level states.

multiloop_fluxonium/two_loop_fluxonium_with_arrays_core.py: Nonlinear array-fluxonium potential/minima utilities in node-flux coordinates.

multiloop_fluxonium/two_loop_fluxonium_with_arrays_normal_modes.py: Linearization near minima (JAX gradient/Hessian) and conversion to oscillator registries/systems.

multiloop_fluxonium/neb.py: Generic NEB / climbing-image barrier finder.

oscillator_integrals/normal_modes.py: Quadratic Hamiltonian diagonalization (EC, Gamma) into normal modes/transforms.

oscillator_integrals/oscillators.py: LocalHarmonicOscillator data model and constructor from circuit matrices.

oscillator_integrals/gaussian_hermite_nd.py: Core ND Gaussian × Hermite integral engine.

oscillator_integrals/oscillator_overlap.py: Precompute/cached overlap internals and overlap engine between two local oscillators.

oscillator_integrals/oscillator_overlap_cache.py: Stateful overlap manager (engine/value cache + Hermitian reuse).

oscillator_integrals/oscillator_operators.py: Matrix elements for charge-like and quadratic operators in Fock basis.

oscillator_integrals/oscillator_system.py: Registry/system convenience layer for named minima and overlap matrices.

oscillator_integrals/oscillator_integrals_1d_quadrature.py: Wrappers around compiled 1D quadrature/Cython routines + Löwdin orthogonalization.

decoherence/quantum_noise.py: Unsymmetrized quantum noise model S(omega) with Bose factor + cutoff options.

decoherence/fgr.py: Generic Fermi Golden Rule rate helper with unit conversion and flexible spectral-density signatures.

3) Core classes/functions
Fluxoid model/system

FluxoidModelParams, FluxoidSector, FluxoidSectorState: parameter/state dataclasses.
TwoLoopFluxoidModel: computes effective flux, offsets, potentials, and builds sector scqubits objects.
FluxoidSectorManager: lazy cache of per-sector eigensystems.
TwoLoopFluxoidSystem: computes cross-sector overlaps in eigenbases; caches overlap blocks by relative shift.
Transition-rate stack

GlobalState: (sector, level) label for global basis.
FluxoidOperator + SectorJumpOperator, SumOperator, ProductOperator, DaggerOperator, ScaledOperator: composable operator DSL.
build_global_states, build_energy_array, build_jump_matrix, build_operator_matrix: vectorized assembly utilities.
compute_rate_matrix, compute_all_decay_rates: full FGR matrix / sparse dict transitions.
thermal_weights, sector_to_sector_rate, sector_rate_matrix(_fast): thermal aggregation from level-space to sector-space.
Array fluxonium

find_minima, node_flux_from_phi_b, potential_energy_node_flux: core nonlinear landscape utilities.
TwoLoopArrayFluxonium: end-to-end minima → Hessian → oscillator(s) → OscillatorRegistry/OscillatorSystem.
FluxoniumMinimum: stores minimum metadata and associated local oscillator.
neb(...): transition-path barrier computation.
Oscillator integral stack

LocalHarmonicOscillator, NormalModeResult, diagonalize_quadratic_hamiltonian(...)
OscillatorPairOverlapData, CachedOverlapInternals, OscillatorOverlapEngine, OverlapManager
integrate_gaussian_hermite(...), G0_tensor(...), apply_L(...)
OperatorMatrixBuilder with overlap_states, n_op, q2_op, q2_total_op
OscillatorRegistry, OscillatorSystem
Decoherence

QuantumNoiseSpectralDensity: J, bose, S, S_array.
fgr_decay_rate(...): scalar FGR transition rate.
calc_therm_ratio(...): (\hbar\omega/(k_B T)) utility.
4) Typical data flow / workflow
Define fluxonium parameters (FluxoidModelParams) and sectors.
Build TwoLoopFluxoidSystem (internally caches per-sector eigensystems and oscillator objects).
Build global state list via build_global_states.
Define transition operator via operator composition (usually sums/products of SectorJumpOperator).
Build operator matrix + energy array (build_operator_matrix, build_energy_array).
Provide spectral density (QuantumNoiseSpectralDensity.S or custom callable).
Compute level-to-level rates (compute_rate_matrix / compute_all_decay_rates).
Optionally thermal-average to sector-to-sector rates (thermal_weights, sector_rate_matrix).
Parallel path for array model:

Use TwoLoopArrayFluxonium to find minima in each ((P_a, P_b)) sector.
Linearize around each minimum (Hessian + capacitance) to produce local oscillators.
Register oscillators and use OscillatorSystem/OperatorMatrixBuilder for overlaps/operator elements.
5) Important design patterns / assumptions
Dataclass-heavy domain modeling for parameters, sectors, minima, and normal-mode outputs.
Cache-first architecture:
sector eigensystems cached by sector key,
overlap matrices cached by relative shift,
oscillator overlap engines/value cache keyed by object identity.
Composite operator pattern (FluxoidOperator) for symbolic construction then matrix realization.
Vectorized block assembly for speed in rate calculations (state-structure indexing by sector).
Active-mode compression in overlap integrals (ignores zero Hermite orders to reduce dimension).
Unit assumptions are strict:
many energies/frequencies treated in GHz-like linear units internally in fluxoid rate code,
FGR converts to angular frequency (2πΔE) and default spectral input is SI rad/s (spectral_omega_units="SI"),
temperatures in Kelvin.
Model assumptions:
core array routines often assume symmetric arrays via N; asymmetric support is only partial/shelved helper paths,
valid quadratic expansion requires physically stable minima (positive-curvature Hessian regions).