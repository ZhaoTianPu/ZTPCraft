"""Matrix-based fluxoid rates with explicit sector/level bookkeeping.

Energies are E/h in GHz; temperatures are kelvin. Operators use [final, initial]
indices and rates use [initial, final]. The spectral density must accept arrays
of SI angular frequencies and a temperature: S(omega, temperature).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from operator import index as integer_index

import numpy as np
from numpy.typing import NDArray
from scipy.constants import h, k

from ztpcraft.decoherence.fgr import rate_matrix_from_spectral_values
from .two_loop_fluxoid_model import FluxoidSector
from .two_loop_fluxoid_system import TwoLoopFluxoidSystem

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SpectralDensity = Callable[[FloatArray, float], FloatArray]


@dataclass(frozen=True)
class SectorBasis:
    """Sector-major basis, retaining levels 0 through nlevels-1 in each sector.

    Missing sector lookups raise KeyError. This basis deliberately does not
    support arbitrary state ordering or unequal level counts.
    """

    sectors: tuple[FluxoidSector, ...]
    nlevels: int
    _positions: dict[FluxoidSector, int] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        sectors = tuple(self.sectors)
        nlevels = integer_index(self.nlevels)
        if not sectors or nlevels < 1:
            raise ValueError("The basis needs at least one sector and one level.")
        positions = {sector: i for i, sector in enumerate(sectors)}
        if len(positions) != len(sectors):
            raise ValueError("Duplicate sectors in the basis.")
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "nlevels", nlevels)
        object.__setattr__(self, "_positions", positions)

    @property
    def nsectors(self) -> int:
        return len(self.sectors)

    @property
    def size(self) -> int:
        return self.nsectors * self.nlevels

    def sector_index(self, sector: FluxoidSector) -> int:
        """Index along a sector axis, e.g. in the aggregated rate matrix."""
        return self._positions[sector]

    def block(self, sector: FluxoidSector) -> slice:
        start = self.sector_index(sector) * self.nlevels
        return slice(start, start + self.nlevels)

    def index(self, sector: FluxoidSector, level: int) -> int:
        level = integer_index(level)
        if not 0 <= level < self.nlevels:
            raise IndexError(level)
        return self.block(sector).start + level

    def label(self, index: int) -> tuple[FluxoidSector, int]:
        index = integer_index(index)
        if not 0 <= index < self.size:
            raise IndexError(index)
        position, level = divmod(index, self.nlevels)
        return self.sectors[position], level


@dataclass(frozen=True)
class RateSetup:
    """A basis, sector energies (GHz), and an operator in that same basis.

    Arrays are copied and made read-only so later changes to an input array do
    not silently change a prepared calculation. Reuse across bath temperatures.
    """

    basis: SectorBasis
    energies: FloatArray
    coupling: ComplexArray

    def __post_init__(self) -> None:
        energies = np.array(self.energies, dtype=float, copy=True)
        coupling = np.array(self.coupling, dtype=complex, copy=True)
        if energies.shape != (self.basis.nsectors, self.basis.nlevels):
            raise ValueError("energies must have shape (nsectors, nlevels).")
        if coupling.shape != (self.basis.size, self.basis.size):
            raise ValueError("coupling must have shape (nstates, nstates).")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(coupling)):
            raise ValueError("Energies and coupling must be finite.")
        energies.setflags(write=False)
        coupling.setflags(write=False)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "coupling", coupling)


def sector_energies(system: TwoLoopFluxoidSystem, basis: SectorBasis) -> FloatArray:
    """Return E/h in GHz, including sector offsets, as [sector, level]."""
    energies = [system.eigenvalues_with_offset(s) for s in basis.sectors]
    if any(len(values) < basis.nlevels for values in energies):
        raise ValueError("The system retains fewer eigenstates than the basis.")
    return np.asarray([values[: basis.nlevels] for values in energies], dtype=float)


def jump_matrix(
    system: TwoLoopFluxoidSystem,
    basis: SectorBasis,
    delta: tuple[int, int],
) -> ComplexArray:
    """Projected jump J[final, initial] for a displacement (dm_a, dm_b).

    Destinations outside the retained basis are projected out. A product of
    projected jumps is generally not the same as a direct combined jump, since
    its intermediate sectors and levels are also truncated.
    """
    dm_a, dm_b = (integer_index(value) for value in delta)
    included = set(basis.sectors)
    matrix = np.zeros((basis.size, basis.size), dtype=complex)
    for source in basis.sectors:
        destination = FluxoidSector(source.m_a + dm_a, source.m_b + dm_b)
        if destination not in included:
            continue
        overlap = system.overlap_matrix(destination, source)
        if min(overlap.shape) < basis.nlevels:
            raise ValueError("The system retains fewer eigenstates than the basis.")
        matrix[basis.block(destination), basis.block(source)] = overlap[
            : basis.nlevels, : basis.nlevels
        ]
    return matrix


def prepare_rates(
    system: TwoLoopFluxoidSystem, basis: SectorBasis, coupling: ComplexArray
) -> RateSetup:
    """Collect fixed spectral data once, after constructing the coupling matrix."""
    return RateSetup(basis, sector_energies(system, basis), coupling)


def calculate_state_rates(
    setup: RateSetup,
    spectral_density: SpectralDensity,
    temperature: float,
    *,
    matrix_element_cutoff: float = 1e-14,
) -> FloatArray:
    """Return Gamma[initial, final] in inverse seconds.

    S must accept an omega array (rad/s) and temperature (K), and return a real,
    nonnegative, broadcastable array in energy-squared times seconds for a
    dimensionless coupling. Use noise.S_array for OhmicLikeNoise. Errors in S
    propagate directly; scalar-only functions require an explicit adapter.
    """
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    energy = setup.energies.ravel()
    omega = 2 * np.pi * 1e9 * (energy[:, None] - energy[None, :])
    spectral = spectral_density(omega, temperature)
    return rate_matrix_from_spectral_values(
        setup.coupling, spectral, matrix_element_cutoff=matrix_element_cutoff
    )


def thermal_populations(energies: FloatArray, temperature: float) -> FloatArray:
    """Conditional Boltzmann probabilities p(level | sector), with E/h in GHz.

    Sector offsets cancel here, but must remain in the energies used for FGR.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Conditional thermal populations require T > 0.")
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or 0 in energies.shape or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a finite, nonempty [sector, level] array.")
    relative = energies - energies.min(axis=1, keepdims=True)
    weights = np.exp(-h * 1e9 * relative / (k * temperature))
    return weights / weights.sum(axis=1, keepdims=True)


def aggregate_rates(
    basis: SectorBasis, state_rates: FloatArray, populations: FloatArray
) -> FloatArray:
    """Average initial levels and sum final levels: Gamma[source, destination].

    The diagonal is zero: intra-sector transitions are not sector jumps. This
    is a rate table, not a Markov generator (whose diagonal is minus the total
    escape rate). Arbitrary normalized conditional populations are supported.
    """
    rates = np.asarray(state_rates, dtype=float)
    populations = np.asarray(populations, dtype=float)
    if rates.shape != (basis.size, basis.size):
        raise ValueError("state_rates must have shape (nstates, nstates).")
    if populations.shape != (basis.nsectors, basis.nlevels):
        raise ValueError("populations must have shape (nsectors, nlevels).")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Rates must be finite and nonnegative.")
    if not np.all(np.isfinite(populations)) or np.any(populations < 0):
        raise ValueError("Populations must be finite and nonnegative.")
    if not np.allclose(populations.sum(axis=1), 1.0):
        raise ValueError("Populations must sum to one within each sector.")
    blocks = rates.reshape(basis.nsectors, basis.nlevels, basis.nsectors, basis.nlevels)
    result = np.einsum("al,albm->ab", populations, blocks)
    np.fill_diagonal(result, 0.0)
    return result


def calculate_sector_rates(
    setup: RateSetup, spectral_density: SpectralDensity, temperature: float
) -> FloatArray:
    """Convenience composition of state rates, populations, and aggregation."""
    populations = thermal_populations(setup.energies, temperature)
    rates = calculate_state_rates(setup, spectral_density, temperature)
    return aggregate_rates(setup.basis, rates, populations)
