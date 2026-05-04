from __future__ import annotations

"""Perturbative first-order hybridization between neighboring phase-slip sectors.

The primary output is a basis transform ``U`` whose columns are dressed states
expressed in the chosen truncated bare basis. Use
:func:`build_perturbative_hybridization_info` to build ``U`` and associated
diagnostics, then pass the result to
:func:`~ztpcraft.projects.fluxonium.multiloop_fluxonium.tunneling_assisted_transition_rates.build_dressed_operator_matrix`
to transform operator matrices.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .tunneling_assisted_model import PhaseSlipSector
from .tunneling_assisted_system import PhaseSlipSystem

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_MATRIX_ELEMENT_CUTOFF = 1e-14


@dataclass(frozen=True)
class GlobalState:
    """Global state label identified by phase-slip sector and in-sector level."""

    sector: PhaseSlipSector
    level: int


@dataclass(frozen=True)
class HybridizationComponent:
    """Single perturbative contribution from a neighboring-sector bare state."""

    source: GlobalState
    partner: GlobalState
    coupling: complex
    detuning: float
    amplitude: complex
    ratio: float
    perturbation_valid: bool
    reason: str | None = None


@dataclass
class PerturbativeHybridizationInfo:
    """Hybridization output in the chosen truncated bare basis.

    ``dressed_eigenvectors_in_bare_basis`` is the basis transform matrix ``U``
    whose columns are dressed states expanded in the chosen bare basis.
    """

    states: list[GlobalState]
    energies: FloatArray
    dressed_eigenvectors_in_bare_basis: ComplexArray
    contributions_by_source: dict[int, list[HybridizationComponent]]
    invalid_contributions: list[HybridizationComponent]


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


def build_perturbative_hybridization_info(
    system: PhaseSlipSystem,
    states: list[GlobalState],
    *,
    perturbation_ratio_threshold: float = 0.2,
    detuning_floor: float = 1e-12,
) -> PerturbativeHybridizationInfo:
    """Build first-order hybridization in a truncated bare basis.

    Parameters
    ----------
    system:
        Phase-slip system providing energies and tunneling matrix elements.
    states:
        Explicit truncated bare basis. Each entry is a
        ``GlobalState(sector=PhaseSlipSector(p), level=mu)``.
    perturbation_ratio_threshold:
        Maximum allowed ``|coupling / detuning|`` before a contribution is
        flagged as invalid.
    detuning_floor:
        Minimum absolute detuning; below this the perturbative amplitude is
        zeroed and the contribution flagged as invalid.

    Returns
    -------
    PerturbativeHybridizationInfo
        Hybridization data including energies, basis transform matrix, and
        per-source perturbative contribution diagnostics.
    """
    if perturbation_ratio_threshold <= 0.0:
        raise ValueError("perturbation_ratio_threshold must be positive.")
    if detuning_floor <= 0.0:
        raise ValueError("detuning_floor must be positive.")

    # Avoid circular import: build_energy_array lives in the transition_rates
    # module which imports from this module in some sweep helpers.
    from .tunneling_assisted_transition_rates import build_energy_array

    basis_states = list(states)
    n_states = len(basis_states)
    energies = build_energy_array(system, basis_states)
    allowed_sectors = {state.sector for state in basis_states}

    state_to_index = {state: idx for idx, state in enumerate(basis_states)}
    # ``U`` is a basis transform from bare -> dressed components: identity on
    # the source-sector component plus dense neighbor-sector blocks from
    # first-order perturbative amplitudes.
    dressed_eigenvectors_in_bare_basis = np.eye(n_states, dtype=np.complex128)
    contributions_by_source: dict[int, list[HybridizationComponent]] = {
        idx: [] for idx in range(n_states)
    }
    invalid_components: list[HybridizationComponent] = []

    for source_idx, source_state in enumerate(basis_states):
        for partner_sector in _neighbor_sectors(source_state.sector, allowed_sectors):
            partner_sector_state = system.get_sector_state(partner_sector)
            for partner_level in range(len(partner_sector_state.evals)):
                partner_state = GlobalState(
                    sector=partner_sector, level=partner_level
                )
                partner_idx = state_to_index.get(partner_state)
                if partner_idx is None:
                    continue

                coupling = system.tunneling_matrix_element(
                    source_state.sector,
                    source_state.level,
                    partner_sector,
                    partner_level,
                )
                detuning = float(energies[source_idx] - energies[partner_idx])

                if abs(detuning) < detuning_floor:
                    amplitude: complex = 0.0j
                    ratio = float("inf")
                    valid = False
                    reason: str | None = (
                        f"|detuning| < detuning_floor ({detuning_floor:.3e})"
                    )
                else:
                    amplitude = coupling / detuning
                    ratio = abs(amplitude)
                    valid = bool(ratio <= perturbation_ratio_threshold)
                    reason = (
                        None if valid else "hybridization_ratio_exceeds_threshold"
                    )

                component = HybridizationComponent(
                    source=source_state,
                    partner=partner_state,
                    coupling=complex(coupling),
                    detuning=detuning,
                    amplitude=complex(amplitude),
                    ratio=float(ratio),
                    perturbation_valid=valid,
                    reason=reason,
                )
                contributions_by_source[source_idx].append(component)
                if not valid:
                    invalid_components.append(component)

                if abs(amplitude) > _MATRIX_ELEMENT_CUTOFF:
                    dressed_eigenvectors_in_bare_basis[
                        partner_idx, source_idx
                    ] += amplitude

    return PerturbativeHybridizationInfo(
        states=basis_states,
        energies=energies,
        dressed_eigenvectors_in_bare_basis=dressed_eigenvectors_in_bare_basis,
        contributions_by_source=contributions_by_source,
        invalid_contributions=invalid_components,
    )
