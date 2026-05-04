from __future__ import annotations

"""Noise channels, operator-matrix construction, and FGR transition-rate
utilities for the phase-slip reduced model.

The typical workflow is:

1. Build a truncated bare basis with :func:`build_global_states`.
2. Build the perturbative hybridization via
   :func:`~ztpcraft.projects.fluxonium.multiloop_fluxonium.tunneling_assisted_hybridization.build_perturbative_hybridization_info`.
3. Define one or more :class:`NoiseChannel`\\s.
4. Compute rates with :func:`rate_matrix_for_channel`,
   :func:`state_to_state_transition_rates_multi_channel`,
   :func:`sector_to_sector_rate`, :func:`sector_rate_matrix`, or
   :func:`neighbor_sector_rates`.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import h, k as k_B

from ztpcraft.decoherence.fgr import FrequencyUnit, compute_rate_matrix

from .tunneling_assisted_hybridization import (
    GlobalState,
    PerturbativeHybridizationInfo,
)
from .tunneling_assisted_model import PhaseSlipSector
from .tunneling_assisted_system import PhaseSlipSystem

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_MATRIX_ELEMENT_CUTOFF = 1e-14

OperatorSelector = str | Callable[[Any], NDArray[Any]]


@dataclass(frozen=True)
class NoiseChannel:
    """Noise channel definition: operator + spectral density + user-facing name."""

    name: str
    operator: OperatorSelector
    spectral_density: Callable[..., float | complex]


@dataclass(frozen=True)
class PhaseSlipJumpOperator:
    """Inter-sector jump operator selecting a fixed phase-slip displacement.

    This is a lightweight analog of
    :class:`~ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_transition_rates.SectorJumpOperator`
    for the phase-slip model. The bare matrix element is the eigenbasis overlap
    ``<partner | source>``; the optional ``include_es`` flag multiplies by the
    phase-slip amplitude ``E_S`` so the same object can serve both as a tunneling
    coupling and as a noise-operator primitive.

    Parameters
    ----------
    system:
        Phase-slip system providing overlap matrices and (if
        ``include_es=True``) the phase-slip amplitude ``E_S``.
    dp:
        Phase-slip sector displacement such that ``partner.p = source.p + dp``.
    include_es:
        If ``True``, multiply the overlap by ``system.model.params.E_S``.
    """

    system: PhaseSlipSystem
    dp: int
    include_es: bool = False

    def matrix_element(self, s1: GlobalState, s2: GlobalState) -> complex:
        """Matrix element ``<s1 | J | s2>`` where ``J`` is this jump operator."""
        if s1.sector.p != s2.sector.p + self.dp:
            return 0.0j
        overlap = self.system.overlap_matrix(s1.sector, s2.sector)
        value = complex(overlap[s1.level, s2.level])
        if self.include_es:
            value *= complex(self.system.model.params.E_S)
        return value


# ---------------------------------------------------------------------------
# Basis / sector helpers
# ---------------------------------------------------------------------------


def ensure_explicit_sector_labels(p_values: Iterable[int]) -> list[int]:
    """Normalize user-provided ``p`` labels (unique + sorted) and validate non-empty.

    Parameters
    ----------
    p_values:
        Iterable of integer-like phase-slip sector labels.

    Returns
    -------
    list[int]
        Sorted unique list of explicit phase-slip sector indices.
    """
    out = sorted(set(int(p) for p in p_values))
    if len(out) == 0:
        raise ValueError("At least one explicit p value is required.")
    return out


def _neighbor_sectors(
    sector: PhaseSlipSector, allowed: set[PhaseSlipSector]
) -> tuple[PhaseSlipSector, ...]:
    """Return nearest-neighbor sectors (``p ± 1``) restricted to ``allowed``."""
    allowed_by_p = {s.p: s for s in allowed}
    neighbors: list[PhaseSlipSector] = []
    for dp in (-1, 1):
        candidate = allowed_by_p.get(sector.p + dp)
        if candidate is not None:
            neighbors.append(candidate)
    return tuple(neighbors)


def _state_structure(
    states: list[GlobalState],
) -> tuple[
    dict[PhaseSlipSector, NDArray[np.int64]],
    dict[PhaseSlipSector, NDArray[np.int64]],
]:
    """Return per-sector index and level lookup arrays."""
    indices: dict[PhaseSlipSector, list[int]] = {}
    levels: dict[PhaseSlipSector, list[int]] = {}
    for idx, state in enumerate(states):
        indices.setdefault(state.sector, []).append(idx)
        levels.setdefault(state.sector, []).append(state.level)
    return (
        {sector: np.asarray(v, dtype=np.int64) for sector, v in indices.items()},
        {sector: np.asarray(v, dtype=np.int64) for sector, v in levels.items()},
    )


def build_global_states(
    sectors: Iterable[PhaseSlipSector], system: PhaseSlipSystem
) -> list[GlobalState]:
    """Enumerate all ``(sector, level)`` states for provided phase-slip sectors.

    Parameters
    ----------
    sectors:
        Iterable of phase-slip sector labels.
    system:
        Phase-slip system used to determine level counts.

    Returns
    -------
    list[GlobalState]
        Flattened list of states in sector-major order.
    """
    states: list[GlobalState] = []
    for sector in sectors:
        evals = system.get_sector_state(sector).evals
        for level in range(len(evals)):
            states.append(GlobalState(sector=sector, level=level))
    return states


def build_energy_array(
    system: PhaseSlipSystem, states: list[GlobalState]
) -> FloatArray:
    """Build energy vector aligned with the ``states`` ordering."""
    sector_indices, sector_levels = _state_structure(states)
    energy_by_sector: dict[PhaseSlipSector, FloatArray] = {}
    energies = np.empty(len(states), dtype=np.float64)
    for sector, indices in sector_indices.items():
        if sector not in energy_by_sector:
            energy_by_sector[sector] = np.asarray(
                system.eigenvalues_with_offset(sector), dtype=np.float64
            )
        energies[indices] = energy_by_sector[sector][sector_levels[sector]]
    return energies


# ---------------------------------------------------------------------------
# Operator matrices (bare + dressed)
# ---------------------------------------------------------------------------


def build_operator_matrix(
    system: PhaseSlipSystem,
    states: list[GlobalState],
    operator: OperatorSelector,
) -> ComplexArray:
    """Build dense bare-basis operator matrix for the global state ordering.

    For ``OperatorSelector`` inputs the operator is local to each sector, so
    the result is block-diagonal over the sectors appearing in ``states``.
    """
    n_states = len(states)
    matrix = np.zeros((n_states, n_states), dtype=np.complex128)
    sector_indices, sector_levels = _state_structure(states)
    for sector, idx in sector_indices.items():
        op_sector = system.sector_operator_in_eigenbasis(sector, operator)
        levels = sector_levels[sector]
        block = op_sector[np.ix_(levels, levels)]
        matrix[np.ix_(idx, idx)] = block
    return matrix


def build_dressed_operator_matrix(
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    operator: OperatorSelector,
    bare_operator_matrix: ComplexArray | None = None,
) -> ComplexArray:
    """Return ``U^dagger O U`` in the same truncated basis as ``hybridization.states``.

    Parameters
    ----------
    system:
        Phase-slip system used to build the bare operator if one is not given.
    hybridization:
        Hybridization result from
        :func:`~ztpcraft.projects.fluxonium.multiloop_fluxonium.tunneling_assisted_hybridization.build_perturbative_hybridization_info`.
    operator:
        Operator selector (scqubits method name or callable on the fluxonium).
    bare_operator_matrix:
        Optional precomputed bare-basis operator matrix matching
        ``hybridization.states``.

    Returns
    -------
    ComplexArray
        Dressed operator matrix in the same basis ordering as
        ``hybridization.states``.
    """
    bare = (
        build_operator_matrix(system, hybridization.states, operator)
        if bare_operator_matrix is None
        else bare_operator_matrix
    )
    U = hybridization.dressed_eigenvectors_in_bare_basis
    return np.asarray(U.conj().T @ bare @ U, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Rate evaluation (single- and multi-channel)
# ---------------------------------------------------------------------------


def rate_matrix_for_channel(
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    channel: NoiseChannel,
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> FloatArray:
    """Compute the full dense transition-rate matrix for one noise channel."""
    dressed_operator = build_dressed_operator_matrix(
        system=system,
        hybridization=hybridization,
        operator=channel.operator,
    )
    return compute_rate_matrix(
        energies=hybridization.energies,
        O_matrix=dressed_operator,
        spectral_density=channel.spectral_density,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )


def rate_matrices_multi_channel(
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    channels: Sequence[NoiseChannel],
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> tuple[dict[str, FloatArray], FloatArray]:
    """Return per-channel dense rate matrices and their sum.

    Returns
    -------
    tuple[dict[str, FloatArray], FloatArray]
        ``(per_channel_rate_matrices, total_rate_matrix)``.
    """
    if len(channels) == 0:
        raise ValueError("channels must be non-empty.")

    seen: set[str] = set()
    for channel in channels:
        if channel.name in seen:
            raise ValueError(f"Duplicate channel name: {channel.name!r}")
        seen.add(channel.name)

    n = len(hybridization.states)
    total = np.zeros((n, n), dtype=np.float64)
    per_channel: dict[str, FloatArray] = {}
    for channel in channels:
        matrix = rate_matrix_for_channel(
            system=system,
            hybridization=hybridization,
            channel=channel,
            T=T,
            units=units,
            spectral_omega_units=spectral_omega_units,
        )
        per_channel[channel.name] = matrix
        total += matrix
    return per_channel, total


def state_to_state_transition_rates_multi_channel(
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    channels: Sequence[NoiseChannel],
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> tuple[dict[str, dict[tuple[int, int], float]], dict[tuple[int, int], float]]:
    """Sparse state-to-state decay-rate dictionaries for multiple channels.

    Returns ``(per_channel, total)`` where each value is a dictionary keyed by
    ``(i, j)`` state-index pairs for positive off-diagonal rates only.
    """
    per_channel_matrices, total_matrix = rate_matrices_multi_channel(
        system=system,
        hybridization=hybridization,
        channels=channels,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )

    per_channel_decay: dict[str, dict[tuple[int, int], float]] = {}
    for name, rate_matrix in per_channel_matrices.items():
        channel_decay: dict[tuple[int, int], float] = {}
        for i in range(rate_matrix.shape[0]):
            for j in range(rate_matrix.shape[1]):
                if i == j or rate_matrix[i, j] <= 0.0:
                    continue
                channel_decay[(i, j)] = float(rate_matrix[i, j])
        per_channel_decay[name] = channel_decay

    total_decay: dict[tuple[int, int], float] = {}
    for i in range(total_matrix.shape[0]):
        for j in range(total_matrix.shape[1]):
            if i == j or total_matrix[i, j] <= 0.0:
                continue
            total_decay[(i, j)] = float(total_matrix[i, j])
    return per_channel_decay, total_decay


# ---------------------------------------------------------------------------
# Thermal weights and sector aggregation
# ---------------------------------------------------------------------------


def thermal_weights(
    system: PhaseSlipSystem, sector: PhaseSlipSector, T: float
) -> FloatArray:
    """Normalized Boltzmann weights within a fixed phase-slip sector."""
    if T <= 0.0:
        raise ValueError("Temperature T must be positive.")
    energies = np.asarray(
        system.eigenvalues_with_offset(sector), dtype=np.float64
    )
    min_energy = float(np.min(energies))
    beta = h / (k_B * T)
    exponents = -beta * (energies - min_energy) * 1e9
    weights = np.exp(exponents)
    partition = float(np.sum(weights))
    if partition <= 0.0:
        raise ValueError("Thermal partition function is non-positive.")
    return np.asarray(weights / partition, dtype=np.float64)


def sector_to_sector_rate(
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    channel: NoiseChannel,
    sector_from: PhaseSlipSector,
    sector_to: PhaseSlipSector,
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> float:
    """Thermally averaged rate from ``sector_from`` to ``sector_to`` for one channel."""
    rates = rate_matrix_for_channel(
        system=system,
        hybridization=hybridization,
        channel=channel,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )
    states = hybridization.states
    indices_from = np.asarray(
        [i for i, s in enumerate(states) if s.sector == sector_from], dtype=np.int64
    )
    indices_to = np.asarray(
        [i for i, s in enumerate(states) if s.sector == sector_to], dtype=np.int64
    )
    if indices_from.size == 0 or indices_to.size == 0:
        return 0.0

    levels_from = np.asarray(
        [states[int(i)].level for i in indices_from], dtype=np.int64
    )
    weights = thermal_weights(system, sector_from, T)[levels_from]
    block = rates[np.ix_(indices_from, indices_to)]
    return float(np.sum(weights[:, None] * block))


def sector_rate_matrix_fast(
    states: list[GlobalState],
    sectors: list[PhaseSlipSector],
    rates: FloatArray,
    weights: dict[PhaseSlipSector, FloatArray],
) -> FloatArray:
    """Aggregate a precomputed state-rate matrix into a sector-rate matrix.

    Parameters
    ----------
    states:
        Ordered global-state basis matching ``rates``.
    sectors:
        Sector order used for the output matrix rows/columns.
    rates:
        Dense state-to-state rate matrix with shape ``(len(states), len(states))``.
    weights:
        Precomputed thermal weights per sector, indexed by in-sector level.

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
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    channel: NoiseChannel,
    sectors: list[PhaseSlipSector],
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> FloatArray:
    """Sector-to-sector thermal rate matrix for one channel.

    Returns the matrix ``R[a, b] = Gamma_{sectors[a] -> sectors[b]}`` where each
    row is thermally averaged over bare levels of the source sector.
    """
    matrix = np.zeros((len(sectors), len(sectors)), dtype=np.float64)
    for i, sector_from in enumerate(sectors):
        for j, sector_to in enumerate(sectors):
            matrix[i, j] = sector_to_sector_rate(
                system=system,
                hybridization=hybridization,
                channel=channel,
                sector_from=sector_from,
                sector_to=sector_to,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
    return matrix


def neighbor_sector_rates(
    system: PhaseSlipSystem,
    hybridization: PerturbativeHybridizationInfo,
    channel: NoiseChannel,
    sectors: Sequence[PhaseSlipSector],
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> dict[PhaseSlipSector, dict[PhaseSlipSector, float]]:
    """Return aggregated nearest-neighbor (``p -> p ± 1``) thermal rates.

    Returns
    -------
    dict[PhaseSlipSector, dict[PhaseSlipSector, float]]
        Nested mapping ``{sector_from: {sector_to: rate}}`` restricted to
        nearest-neighbor targets present in ``sectors``.
    """
    allowed = set(sectors)
    out: dict[PhaseSlipSector, dict[PhaseSlipSector, float]] = {}
    for sector in sectors:
        out[sector] = {}
        for target in _neighbor_sectors(sector, allowed):
            out[sector][target] = sector_to_sector_rate(
                system=system,
                hybridization=hybridization,
                channel=channel,
                sector_from=sector,
                sector_to=target,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
    return out
