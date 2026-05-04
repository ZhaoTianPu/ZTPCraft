from __future__ import annotations

"""Operator construction and FGR transition-rate utilities for fluxoid sectors."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import h, k as k_B

from ztpcraft.decoherence.fgr import FrequencyUnit, compute_rate_matrix
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_MATRIX_ELEMENT_CUTOFF = 1e-14


@dataclass(frozen=True)
class GlobalState:
    """Global state label identified by sector and in-sector level index."""

    sector: FluxoidSector
    level: int


def _shift_sector(sector: FluxoidSector, delta: tuple[int, int]) -> FluxoidSector:
    """Return sector shifted by integer delta in `(m_a, m_b)`."""
    return FluxoidSector(sector.m_a + delta[0], sector.m_b + delta[1])


def _sector_key(sector: FluxoidSector) -> tuple[int, int]:
    """Return class-agnostic hash/equality key for a fluxoid sector."""
    return (sector.m_a, sector.m_b)


def build_global_states(
    sectors: Iterable[FluxoidSector], system: TwoLoopFluxoidSystem
) -> list[GlobalState]:
    """Enumerate all `(sector, level)` states for provided sectors.

    Parameters
    ----------
    sectors:
        Iterable of sector labels.
    system:
        Fluxoid system object used to determine level counts.

    Returns
    -------
    list[GlobalState]
        Flattened list of states in sector-major order.
    """
    states: list[GlobalState] = []
    for sector in sectors:
        evals = system.eigenvalues_with_offset(sector)
        for level in range(len(evals)):
            states.append(GlobalState(sector=sector, level=level))
    return states


class FluxoidOperator(ABC):
    """Abstract operator over `GlobalState` basis with composable algebra."""

    def __add__(self, other: "FluxoidOperator") -> "FluxoidOperator":
        return SumOperator(self, other)

    def __mul__(self, other: "FluxoidOperator") -> "FluxoidOperator":
        return ProductOperator(self, other)

    def __rmul__(self, scalar: float | complex) -> "FluxoidOperator":
        return ScaledOperator(complex(scalar), self)

    def dagger(self) -> "FluxoidOperator":
        return DaggerOperator(self)

    def bind_states(self, states: list[GlobalState]) -> "FluxoidOperator":
        return self

    @abstractmethod
    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        raise NotImplementedError


@dataclass(frozen=True)
class SectorJumpOperator(FluxoidOperator):
    """Inter-sector jump operator selecting a fixed sector displacement."""

    system: TwoLoopFluxoidSystem
    delta: tuple[int, int]

    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        shifted = _shift_sector(s2.sector, self.delta)
        if _sector_key(s1.sector) != _sector_key(shifted):
            return 0.0j
        overlap = self.system.overlap_matrix(s1.sector, s2.sector)
        return complex(overlap[s1.level, s2.level])


@dataclass(frozen=True)
class SumOperator(FluxoidOperator):
    """Operator sum `left + right`."""

    left: FluxoidOperator
    right: FluxoidOperator

    def bind_states(self, states: list[GlobalState]) -> "FluxoidOperator":
        return SumOperator(
            self.left.bind_states(states), self.right.bind_states(states)
        )

    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        return self.left.matrix_element(s1, s2) + self.right.matrix_element(s1, s2)


@dataclass(frozen=True)
class ProductOperator(FluxoidOperator):
    """Operator product `left @ right` evaluated in a bound basis."""

    left: FluxoidOperator
    right: FluxoidOperator
    states: tuple[GlobalState, ...] | None = None

    def bind_states(self, states: list[GlobalState]) -> "FluxoidOperator":
        return ProductOperator(
            self.left.bind_states(states),
            self.right.bind_states(states),
            tuple(states),
        )

    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        if self.states is None:
            raise ValueError("ProductOperator requires bound global states.")
        total = 0.0j
        for state_k in self.states:
            total += self.left.matrix_element(s1, state_k) * self.right.matrix_element(
                state_k, s2
            )
        return total


@dataclass(frozen=True)
class DaggerOperator(FluxoidOperator):
    """Hermitian-conjugate wrapper for another operator."""

    operator: FluxoidOperator

    def bind_states(self, states: list[GlobalState]) -> "FluxoidOperator":
        return DaggerOperator(self.operator.bind_states(states))

    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        return np.conj(self.operator.matrix_element(s2, s1))


@dataclass(frozen=True)
class ScaledOperator(FluxoidOperator):
    """Scalar-multiplied operator."""

    scalar: complex
    operator: FluxoidOperator

    def bind_states(self, states: list[GlobalState]) -> "FluxoidOperator":
        return ScaledOperator(self.scalar, self.operator.bind_states(states))

    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        return self.scalar * self.operator.matrix_element(s1, s2)


def _state_structure(
    states: list[GlobalState],
) -> tuple[
    dict[FluxoidSector, NDArray[np.int64]], dict[FluxoidSector, NDArray[np.int64]]
]:
    """Return per-sector index and level lookup arrays."""
    indices: dict[FluxoidSector, list[int]] = {}
    levels: dict[FluxoidSector, list[int]] = {}
    for idx, state in enumerate(states):
        indices.setdefault(state.sector, []).append(idx)
        levels.setdefault(state.sector, []).append(state.level)
    return (
        {sector: np.asarray(v, dtype=np.int64) for sector, v in indices.items()},
        {sector: np.asarray(v, dtype=np.int64) for sector, v in levels.items()},
    )


def build_energy_array(
    system: TwoLoopFluxoidSystem, states: list[GlobalState]
) -> FloatArray:
    """Build energy vector aligned with `states` ordering.

    Parameters
    ----------
    system:
        Fluxoid system object.
    states:
        Ordered list of global states.

    Returns
    -------
    FloatArray
        Energies in the same order as `states`.
    """
    sector_indices, sector_levels = _state_structure(states)
    energy_by_sector: dict[FluxoidSector, FloatArray] = {}
    energies = np.empty(len(states), dtype=np.float64)
    for sector, indices in sector_indices.items():
        if sector not in energy_by_sector:
            energy_by_sector[sector] = np.asarray(
                system.eigenvalues_with_offset(sector), dtype=np.float64
            )
        energies[indices] = energy_by_sector[sector][sector_levels[sector]]
    return energies


def build_jump_matrix(
    system: TwoLoopFluxoidSystem,
    states: list[GlobalState],
    delta: tuple[int, int],
    overlap_cache: (
        dict[tuple[FluxoidSector, FluxoidSector], ComplexArray] | None
    ) = None,
) -> ComplexArray:
    """Construct dense jump matrix for a fixed sector displacement.

    Parameters
    ----------
    system:
        Fluxoid system providing overlap matrices.
    states:
        Ordered global-state basis.
    delta:
        Sector displacement `(dm_a, dm_b)`.
    overlap_cache:
        Optional overlap cache reused across calls.

    Returns
    -------
    ComplexArray
        Dense jump operator matrix in the global basis.
    """
    n_states = len(states)
    matrix = np.zeros((n_states, n_states), dtype=np.complex128)
    sector_indices, sector_levels = _state_structure(states)
    sector_by_key = {_sector_key(sector): sector for sector in sector_indices}
    indices_by_key = {
        _sector_key(sector): indices for sector, indices in sector_indices.items()
    }
    levels_by_key = {
        _sector_key(sector): levels for sector, levels in sector_levels.items()
    }
    cache = overlap_cache if overlap_cache is not None else {}

    for sector_src, idx_src in sector_indices.items():
        shifted_dst = _shift_sector(sector_src, delta)
        sector_dst_key = _sector_key(shifted_dst)
        sector_dst = sector_by_key.get(sector_dst_key)
        if sector_dst is None:
            continue
        idx_dst = indices_by_key.get(sector_dst_key)
        if idx_dst is None:
            continue

        key = (sector_dst, sector_src)
        overlap = cache.get(key)
        if overlap is None:
            overlap = np.asarray(
                system.overlap_matrix(sector_dst, sector_src), dtype=np.complex128
            )
            cache[key] = overlap

        lvl_dst = levels_by_key[sector_dst_key]
        lvl_src = sector_levels[sector_src]
        block = overlap[np.ix_(lvl_dst, lvl_src)]
        matrix[np.ix_(idx_dst, idx_src)] = block
    return matrix


def build_operator_matrix(
    operator: FluxoidOperator,
    system: TwoLoopFluxoidSystem,
    states: list[GlobalState],
) -> ComplexArray:
    """Build dense matrix for composite `FluxoidOperator` in global basis.

    Parameters
    ----------
    operator:
        Operator expression built from `FluxoidOperator` primitives.
    system:
        Fluxoid system used by jump operators.
    states:
        Ordered global-state basis.

    Returns
    -------
    ComplexArray
        Dense operator matrix for the provided basis.
    """
    overlap_cache: dict[tuple[FluxoidSector, FluxoidSector], ComplexArray] = {}
    jump_cache: dict[tuple[int, int], ComplexArray] = {}
    op_cache: dict[int, ComplexArray] = {}

    def _build(op: FluxoidOperator) -> ComplexArray:
        op_id = id(op)
        cached = op_cache.get(op_id)
        if cached is not None:
            return cached

        out: ComplexArray
        if isinstance(op, SectorJumpOperator):
            if op.system is not system:
                raise ValueError(
                    "SectorJumpOperator.system must match the provided system."
                )
            key = (op.delta[0], op.delta[1])
            cached_jump = jump_cache.get(key)
            if cached_jump is None:
                cached_jump = build_jump_matrix(
                    system=system,
                    states=states,
                    delta=op.delta,
                    overlap_cache=overlap_cache,
                )
                jump_cache[key] = cached_jump
            out = cached_jump
        elif isinstance(op, SumOperator):
            out = _build(op.left) + _build(op.right)
        elif isinstance(op, ProductOperator):
            out = _build(op.left) @ _build(op.right)
        elif isinstance(op, DaggerOperator):
            out = _build(op.operator).conj().T
        elif isinstance(op, ScaledOperator):
            out = op.scalar * _build(op.operator)
        else:
            raise TypeError(f"Unsupported operator type: {type(op).__name__}")

        op_cache[op_id] = out
        return out

    return _build(operator)


def compute_all_decay_rates(
    system: TwoLoopFluxoidSystem,
    states: list[GlobalState],
    operator: FluxoidOperator,
    spectral_density: Callable[..., float | complex],
    T: float | None,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> dict[tuple[int, int], float]:
    """Return sparse decay-rate dictionary for non-negligible transitions.

    Parameters
    ----------
    system:
        Fluxoid system object.
    states:
        Ordered global-state basis.
    operator:
        Operator expression used in FGR rates.
    spectral_density:
        Spectral density callable `S(omega)` or `S(omega, T)`.
    T:
        Temperature in Kelvin or `None`.
    units:
        Unit used by energies.
    spectral_omega_units:
        Units expected by spectral-density callable.

    Returns
    -------
    dict[tuple[int, int], float]
        Sparse mapping from transition index pairs `(i, j)` to rates.
    """
    energies = build_energy_array(system, states)
    operator_matrix = build_operator_matrix(operator, system, states)
    rate_matrix = compute_rate_matrix(
        energies=energies,
        O_matrix=operator_matrix,
        spectral_density=spectral_density,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )
    rates: dict[tuple[int, int], float] = {}
    n_states = len(states)
    mask = np.abs(operator_matrix) >= _MATRIX_ELEMENT_CUTOFF
    np.fill_diagonal(mask, False)
    idx_pairs = np.argwhere(mask)
    for i, j in idx_pairs:
        if i < 0 or j < 0 or i >= n_states or j >= n_states:
            continue
        rates[(int(i), int(j))] = float(rate_matrix[i, j])
    return rates


def thermal_weights(
    system: TwoLoopFluxoidSystem, sector: FluxoidSector, T: float
) -> FloatArray:
    """Compute normalized Boltzmann weights for one sector.

    Parameters
    ----------
    system:
        Fluxoid system object.
    sector:
        Sector for which thermal weights are computed.
    T:
        Temperature in Kelvin.

    Returns
    -------
    FloatArray
        Normalized Boltzmann weights over levels in `sector`.
    """
    if T <= 0.0:
        raise ValueError("Temperature T must be positive.")

    energies = np.asarray(system.eigenvalues_with_offset(sector), dtype=np.float64)
    min_energy = float(np.min(energies))
    beta = h / (k_B * T)
    exponents = -beta * (energies - min_energy) * 1e9
    weights = np.exp(exponents)
    partition = float(np.sum(weights))
    if partition <= 0.0:
        raise ValueError("Thermal partition function is non-positive.")
    return np.asarray(weights / partition, dtype=np.float64)


def sector_to_sector_rate(
    system: TwoLoopFluxoidSystem,
    sector_from: FluxoidSector,
    sector_to: FluxoidSector,
    states: list[GlobalState],
    operator: FluxoidOperator,
    spectral_density: Callable[..., float | complex],
    T: float,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> float:
    """Compute thermalized aggregate rate from one sector to another.

    Parameters
    ----------
    system:
        Fluxoid system object.
    sector_from:
        Initial sector.
    sector_to:
        Final sector.
    states:
        Ordered global-state basis.
    operator:
        Operator expression used in FGR rates.
    spectral_density:
        Spectral density callable.
    T:
        Temperature in Kelvin.
    units:
        Unit used by energies.
    spectral_omega_units:
        Units expected by spectral-density callable.

    Returns
    -------
    float
        Thermalized aggregate transition rate from `sector_from` to `sector_to`.
    """
    energies = build_energy_array(system, states)
    operator_matrix = build_operator_matrix(operator, system, states)
    rates = compute_rate_matrix(
        energies,
        operator_matrix,
        spectral_density,
        T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )
    sector_indices, sector_levels = _state_structure(states)
    idx_from = sector_indices.get(sector_from)
    idx_to = sector_indices.get(sector_to)
    if idx_from is None or idx_to is None or idx_from.size == 0 or idx_to.size == 0:
        return 0.0

    weights = thermal_weights(system, sector_from, T)[sector_levels[sector_from]]
    block = rates[np.ix_(idx_from, idx_to)]
    return float(np.sum(weights[:, None] * block))


def sector_rate_matrix_fast(
    states: list[GlobalState],
    sectors: list[FluxoidSector],
    rates: FloatArray,
    weights: dict[FluxoidSector, FloatArray],
) -> FloatArray:
    """Aggregate precomputed state-rate matrix into sector-rate matrix.

    Parameters
    ----------
    states:
        Ordered global-state basis.
    sectors:
        Sector order used for matrix rows/columns.
    rates:
        Dense state-to-state rate matrix.
    weights:
        Precomputed thermal weights per sector.

    Returns
    -------
    FloatArray
        Sector-to-sector aggregate rate matrix.
    """
    sector_indices, sector_levels = _state_structure(states)
    matrix = np.zeros((len(sectors), len(sectors)), dtype=np.float64)

    for a_idx, sector_from in enumerate(sectors):
        idx_from = sector_indices.get(sector_from)
        if idx_from is None or idx_from.size == 0:
            continue
        w_sector = weights[sector_from]
        w_from = w_sector[sector_levels[sector_from]]
        for b_idx, sector_to in enumerate(sectors):
            idx_to = sector_indices.get(sector_to)
            if idx_to is None or idx_to.size == 0:
                continue
            block = rates[np.ix_(idx_from, idx_to)]
            matrix[a_idx, b_idx] = float(np.sum(w_from[:, None] * block))
    return matrix


def sector_rate_matrix(
    system: TwoLoopFluxoidSystem,
    sectors: list[FluxoidSector],
    states: list[GlobalState],
    operator: FluxoidOperator,
    spectral_density: Callable[..., float | complex],
    T: float,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> FloatArray:
    """Convenience wrapper building full sector-to-sector rate matrix.

    Parameters
    ----------
    system:
        Fluxoid system object.
    sectors:
        Sector order used for matrix rows/columns.
    states:
        Ordered global-state basis.
    operator:
        Operator expression used in FGR rates.
    spectral_density:
        Spectral density callable.
    T:
        Temperature in Kelvin.
    units:
        Unit used by energies.
    spectral_omega_units:
        Units expected by spectral-density callable.

    Returns
    -------
    FloatArray
        Sector-to-sector aggregate rate matrix.
    """
    energies = build_energy_array(system, states)
    operator_matrix = build_operator_matrix(operator, system, states)
    rates = compute_rate_matrix(
        energies,
        operator_matrix,
        spectral_density,
        T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )
    thermal_cache = {sector: thermal_weights(system, sector, T) for sector in sectors}
    return sector_rate_matrix_fast(
        states=states,
        sectors=sectors,
        rates=rates,
        weights=thermal_cache,
    )
