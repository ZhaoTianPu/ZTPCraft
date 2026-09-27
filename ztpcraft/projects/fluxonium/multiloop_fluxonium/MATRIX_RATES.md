# Matrix-based fluxoid rates

The preferred API uses one explicit `SectorBasis` and ordinary NumPy matrices.
The system still caches eigenstates and oscillator overlaps. No operator expression
classes, binding step, or operator factory are needed for a new calculation.

```python
import numpy as np
import ztpcraft.projects.fluxonium.multiloop_fluxonium as mlfx
from ztpcraft.decoherence import OhmicLikeNoise

params = mlfx.FluxoidModelParams(
    EL_a=0.46307, EL_b=0.43693, EJ=5.1, EC=0.812,
    phi_ext_a=0.0, phi_ext_b=0.0, flux_allocation_alpha=1.0,
)
basis = mlfx.SectorBasis(
    tuple(mlfx.FluxoidSector(m, 0) for m in range(-3, 4)), nlevels=10
)
system = mlfx.TwoLoopFluxoidSystem(params, cutoff=60, evals_count=basis.nlevels)
J = mlfx.jump_matrix(system, basis, (1, 0))
setup = mlfx.prepare_rates(system, basis, J + J.conj().T)
noise = OhmicLikeNoise(alpha=1e-80, s=1)

rates = mlfx.calculate_state_rates(setup, noise.S_array, temperature=0.14)
populations = mlfx.thermal_populations(setup.energies, temperature=0.14)
sector_rates = mlfx.aggregate_rates(basis, rates, populations)
# Equivalent shortcut if only sector rates are needed:
sector_rates = mlfx.calculate_sector_rates(setup, noise.S_array, temperature=0.14)
```

The example cutoffs are illustrative, not a convergence claim.

## Array conventions

* `setup.energies[sector, level]`: E/h in GHz, **including sector offsets**.
* `setup.coupling[final, initial]`: dimensionless operator matrix.
* `rates[initial, final]`: FGR transition rate in inverse seconds.
* `populations[sector, level]`: probabilities normalized within each sector.
* `sector_rates[source, destination]`: conditional thermal switching rate.

`basis.block(sector)`, `basis.index(sector, level)`, `basis.label(index)`, and
`basis.sector_index(sector)` provide the bookkeeping. Missing sectors raise
`KeyError`. Each sector retains the same number of low-lying eigenstates.

`RateSetup` copies its arrays and makes them read-only. Reuse it for temperatures
and noise models. Create a new setup for changed energies or couplings; never
mutate a system's parameters after its eigensystems have been cached.

The array-only noise contract is `S(omega_array, temperature)`, with omega in
rad/s, temperature in kelvin, and S in energy-squared times seconds for a
dimensionless coupling. A scalar return broadcasts as white noise. Custom noise
errors propagate without retry. Scalar-only noise can be adapted explicitly:

```python
S_array = np.vectorize(scalar_noise, otypes=[float])
```

The conditional thermal weights average over source levels only; destination
levels are summed. The sector-rate diagonal is zero; a Markov generator requires
a negative escape-rate diagonal instead. `aggregate_rates` also accepts explicit
nonthermal conditional populations, if that is the desired model.

## Operators, baths, and truncation

Use `+`, scalar multiplication, `@`, and `.conj().T` for matrix algebra. Jumps
outside the retained sectors are projected out. Products project intermediate
states too, so a product of two jumps need not equal a direct combined-displacement
jump in a truncated basis. With only `(m_a, 0)` sectors, a B jump is zero.

For independent baths, compute a rate matrix for each bath/coupling and add the
rates. For coherent contributions to one bath coupling, add the matrices before
computing rates. These are physically different operations.

`flux_allocation_alpha` retains the existing overlap prescription. It affects jump
overlaps without affecting spectra. Do not assume it is a harmless gauge parameter
without transforming states and coupling operators consistently.

## Sweeps

For arbitrary flux trajectories, build each point's system, coupling and setup,
then evaluate all temperatures using that setup. Store numeric arrays with axes
`[temperature, flux, source_sector, destination_sector]`, accompanied by the
temperature/flux coordinates and sector labels. No special sweep-result class is
required. Calculate each state-rate matrix once if saving both state and sector
results.

For the particular trajectory `phi_d=(f_b-f_a)*phi_c-2*phi_bias`, each sector's
effective flux and overlaps remain constant. Reuse the reference coupling and
replace energies by reference energies plus the changes in sector offsets. The
notebook demonstrates this optimization; do not apply it to general trajectories.

## Migration and behavior corrections

Old `SectorJumpOperator`, `FluxoidOperator`, and workspace/sweep entry points
remain available for existing callers. The new notebook uses only the matrix
workflow. There is no need to rewrite a historical notebook merely to import the
package, but two correctness changes affect both workflows:

1. `compute_rate_matrix` now consistently uses `O[final, initial]` and returns
   `rates[initial, final]`. Previously it used the opposite matrix element, hidden
   when the coupling was Hermitian. Sparse wrappers use the transposed operator
   mask consistently. Recheck old calculations using non-Hermitian couplings.
2. At finite temperature the Ohmic `s=1` spectrum now uses its finite zero-frequency
   limit `alpha*k*T/hbar`. The array implementation is vectorized and numerically
   stable near zero and at large frequencies. Super-Ohmic `s>1` has zero limit;
   sub-Ohmic `0<s<1` has an infinite limit and needs physical infrared
   regularization before a finite-rate calculation.

The generic legacy FGR callable adapter remains available; the matrix fluxoid API
bypasses its signature inspection and broad scalar fallback. The new shared
`rate_matrix_from_spectral_values` validates matrix/PSD dimensions, finiteness and
nonnegativity, so malformed physics does not silently turn into a plausible plot.

Use portable arrays and JSON metadata for collaboration. Record the exact package
revision, basis, units, noise parameters, trajectory, and any experimental
calibration or display scaling; do not rely on pickled system objects as the
primary exchange format.
