from __future__ import annotations

"""Parameter sweeps for two-loop fluxoid FGR rates.

Diagonalization and global operator-matrix assembly are expensive. For a fixed
external-flux configuration, :func:`build_fgr_workspace` caches energies and the
dense operator matrix so temperature and spectral-density variations only
re-evaluate ``S(ω)`` (and thermal weights for sector aggregation).
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ztpcraft.decoherence.fgr import FrequencyUnit
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidModelParams,
    FluxoidSector,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_transition_rates import (
    FluxoidOperator,
    GlobalState,
    build_energy_array,
    build_global_states,
    build_operator_matrix,
    compute_all_decay_rates,
    compute_rate_matrix,
    sector_rate_matrix,
    sector_rate_matrix_fast,
    thermal_weights,
)

FloatArray = NDArray[np.float64]

_MATRIX_ELEMENT_CUTOFF = 1e-14


@dataclass(frozen=True)
class FluxoidFGRWorkspace:
    """Cached energies and operator matrix for cheap T / spectral sweeps.

    Build with :func:`build_fgr_workspace` using the same ``system`` you will
    pass to sector-rate helpers (for thermal weights). Matrix elements and
    energies are fixed; only bath correlations and temperature enter afterward.
    """

    system: TwoLoopFluxoidSystem
    states: tuple[GlobalState, ...]
    energies: FloatArray
    operator_matrix: ComplexArray

    @property
    def state_list(self) -> list[GlobalState]:
        return list(self.states)


def build_fgr_workspace(
    system: TwoLoopFluxoidSystem,
    states: Sequence[GlobalState],
    operator: FluxoidOperator,
) -> FluxoidFGRWorkspace:
    """Diagonalization-free cache for repeated FGR evaluations.

    Parameters
    ----------
    system:
        Fluxoid system used to build ``operator`` (e.g. ``SectorJumpOperator``).
    states:
        Global-state basis, same order as in overlap / rate routines.
    operator:
        Composite fluxoid operator (bound or unbound; binding is internal).

    Returns
    -------
    FluxoidFGRWorkspace
        Cached ``energies`` and dense ``operator_matrix``.
    """
    state_list = list(states)
    energies = build_energy_array(system, state_list)
    operator_matrix = build_operator_matrix(operator, system, state_list)
    return FluxoidFGRWorkspace(
        system=system,
        states=tuple(state_list),
        energies=energies,
        operator_matrix=operator_matrix,
    )


def rate_matrix_from_workspace(
    workspace: FluxoidFGRWorkspace,
    spectral_density: Callable[..., float | complex],
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> FloatArray:
    """FGR rate matrix using cached energies and operator matrix."""
    return compute_rate_matrix(
        energies=workspace.energies,
        O_matrix=workspace.operator_matrix,
        spectral_density=spectral_density,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )


def sparse_decay_rates_from_workspace(
    workspace: FluxoidFGRWorkspace,
    spectral_density: Callable[..., float | complex],
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> dict[tuple[int, int], float]:
    """Sparse transition dict; same semantics as :func:`compute_all_decay_rates`."""
    rate_matrix = rate_matrix_from_workspace(
        workspace,
        spectral_density,
        T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )
    O = workspace.operator_matrix
    rates: dict[tuple[int, int], float] = {}
    mask = np.abs(O) >= _MATRIX_ELEMENT_CUTOFF
    np.fill_diagonal(mask, False)
    for i, j in np.argwhere(mask):
        rates[(int(i), int(j))] = float(rate_matrix[int(i), int(j)])
    return rates


def sector_rate_matrix_from_workspace(
    workspace: FluxoidFGRWorkspace,
    sectors: list[FluxoidSector],
    spectral_density: Callable[..., float | complex],
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> FloatArray:
    """Sector-aggregated rates; same as :func:`sector_rate_matrix` without rebuilding O."""
    if T <= 0.0:
        raise ValueError("Temperature T must be positive for sector aggregation.")
    rates = rate_matrix_from_workspace(
        workspace,
        spectral_density,
        T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )
    thermal_cache = {
        sector: thermal_weights(workspace.system, sector, T) for sector in sectors
    }
    return sector_rate_matrix_fast(
        states=workspace.state_list,
        sectors=sectors,
        rates=rates,
        weights=thermal_cache,
    )


def sweep_temperatures(
    workspace: FluxoidFGRWorkspace,
    sectors: list[FluxoidSector],
    temperatures: Sequence[float],
    spectral_density: Callable[..., float | complex],
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    include_sparse: bool = False,
) -> list[FloatArray] | tuple[list[FloatArray], list[dict[tuple[int, int], float]]]:
    """Sector rate matrix at each temperature; operator matrix built once.

    Parameters
    ----------
    workspace:
        Precomputed FGR workspace for one flux configuration.
    sectors:
        Row/column order for the sector matrix.
    temperatures:
        Kelvin values (each must be ``> 0`` for thermal weights).
    spectral_density:
        ``S(ω)`` or ``S(ω, T)`` as elsewhere in this package.
    include_sparse:
        If True, also return sparse state-pair rates per temperature.

    Returns
    -------
    list[FloatArray]
        Sector matrices, or ``(sector_matrices, sparse_rates)`` if
        ``include_sparse`` is True.
    """
    sector_mats: list[FloatArray] = []
    sparse_list: list[dict[tuple[int, int], float]] = []
    for T in temperatures:
        gamma = sector_rate_matrix_from_workspace(
            workspace,
            sectors,
            spectral_density,
            T,
            units=units,
            spectral_omega_units=spectral_omega_units,
        )
        sector_mats.append(gamma)
        if include_sparse:
            sparse_list.append(
                sparse_decay_rates_from_workspace(
                    workspace,
                    spectral_density,
                    T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
            )
    if include_sparse:
        return sector_mats, sparse_list
    return sector_mats


def sweep_spectral_densities(
    workspace: FluxoidFGRWorkspace,
    sectors: list[FluxoidSector],
    spectral_models: Sequence[Callable[..., float | complex]],
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    include_sparse: bool = False,
) -> list[FloatArray] | tuple[list[FloatArray], list[dict[tuple[int, int], float]]]:
    """Evaluate several spectral-density callables at fixed temperature and flux.

    Use this when parameters enter only through ``S`` (e.g. amplitude, cutoff
    frequency) so Hamiltonian overlaps are reused.
    """
    sector_mats: list[FloatArray] = []
    sparse_list: list[dict[tuple[int, int], float]] = []
    for S in spectral_models:
        gamma = sector_rate_matrix_from_workspace(
            workspace,
            sectors,
            S,
            T,
            units=units,
            spectral_omega_units=spectral_omega_units,
        )
        sector_mats.append(gamma)
        if include_sparse:
            sparse_list.append(
                sparse_decay_rates_from_workspace(
                    workspace,
                    S,
                    T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
            )
    if include_sparse:
        return sector_mats, sparse_list
    return sector_mats


def with_common_mode_flux(
    base: FluxoidModelParams,
    phi_cm: float,
) -> FluxoidModelParams:
    """Return params with ``phi_cm/2`` added to both ``phi_ext_a`` and ``phi_ext_b``."""
    return FluxoidModelParams(
        EL_a=base.EL_a,
        EL_b=base.EL_b,
        EJ=base.EJ,
        EC=base.EC,
        phi_ext_a=base.phi_ext_a + phi_cm / 2.0,
        phi_ext_b=base.phi_ext_b + phi_cm / 2.0,
        flux_allocation_alpha=base.flux_allocation_alpha,
    )


@dataclass(frozen=True)
class FluxoidFluxSweepResult:
    """One external-flux sample with system, basis, and optional FGR workspace."""

    label: object
    params: FluxoidModelParams
    system: TwoLoopFluxoidSystem
    states: tuple[GlobalState, ...]
    workspace: FluxoidFGRWorkspace | None = None


def sweep_external_flux_transition_setup(
    *,
    sectors: Iterable[FluxoidSector],
    flux_labels_and_params: Sequence[tuple[object, FluxoidModelParams]],
    operator_factory: Callable[[TwoLoopFluxoidSystem], FluxoidOperator],
    cutoff: int = 120,
    evals_count: int = 10,
    build_workspace: bool = True,
) -> list[FluxoidFluxSweepResult]:
    """Build a new :class:`TwoLoopFluxoidSystem` per flux point (full diagonalization).

    Parameters
    ----------
    flux_labels_and_params:
        ``(label, FluxoidModelParams)`` per sample; ``label`` is stored on the
        result (e.g. ``phi_cm`` float or an index).
    sectors:
        Fluxoid sectors included in the global Hilbert space.
    operator_factory:
        Builds the jump / noise operator for each new ``system`` (e.g.
        ``lambda sys: SectorJumpOperator(sys, (1,0)) + SectorJumpOperator(sys, (-1,0))``).
    cutoff, evals_count:
        Passed to :class:`TwoLoopFluxoidSystem`.
    build_workspace:
        If True, precompute :class:`FluxoidFGRWorkspace` for T / spectral sweeps
        at each flux without rebuilding the operator matrix.

    Returns
    -------
    list[FluxoidFluxSweepResult]
        One entry per flux point.
    """
    results: list[FluxoidFluxSweepResult] = []
    sector_list = list(sectors)
    for label, params in flux_labels_and_params:
        system = TwoLoopFluxoidSystem(params, cutoff=cutoff, evals_count=evals_count)
        states = build_global_states(sector_list, system)
        operator = operator_factory(system)
        workspace: FluxoidFGRWorkspace | None = None
        if build_workspace:
            workspace = build_fgr_workspace(system, states, operator)
        results.append(
            FluxoidFluxSweepResult(
                label=label,
                params=params,
                system=system,
                states=tuple(states),
                workspace=workspace,
            )
        )
    return results


def sweep_common_mode_flux_transition_setup(
    base_params: FluxoidModelParams,
    phi_cm_values: Sequence[float],
    sectors: Iterable[FluxoidSector],
    operator_factory: Callable[[TwoLoopFluxoidSystem], FluxoidOperator],
    *,
    cutoff: int = 120,
    evals_count: int = 10,
    build_workspace: bool = True,
) -> list[FluxoidFluxSweepResult]:
    """Convenience wrapper: ``params = with_common_mode_flux(base, phi_cm)`` per point."""
    flux_points = [(phi_cm, with_common_mode_flux(base_params, phi_cm)) for phi_cm in phi_cm_values]
    return sweep_external_flux_transition_setup(
        sectors=sectors,
        flux_labels_and_params=flux_points,
        operator_factory=operator_factory,
        cutoff=cutoff,
        evals_count=evals_count,
        build_workspace=build_workspace,
    )


def rates_at_flux_points(
    flux_results: Sequence[FluxoidFluxSweepResult],
    sectors: list[FluxoidSector],
    spectral_density: Callable[..., float | complex],
    T: float,
    *,
    operator_factory: Callable[[TwoLoopFluxoidSystem], FluxoidOperator] | None = None,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    sparse: bool = False,
) -> tuple[list[FloatArray], list[dict[tuple[int, int], float]] | None]:
    """Compute sector (and optionally sparse) rates for pre-built flux sweep results.

    If results were created with ``build_workspace=True``, uses the cached
    operator matrix. If ``workspace`` is missing, pass ``operator_factory`` to
    rebuild ``O`` once per flux point (still no extra work versus a manual loop).
    """
    sector_mats: list[FloatArray] = []
    sparse_out: list[dict[tuple[int, int], float]] | None = [] if sparse else None
    for entry in flux_results:
        if entry.workspace is not None:
            gamma = sector_rate_matrix_from_workspace(
                entry.workspace,
                sectors,
                spectral_density,
                T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
            sector_mats.append(gamma)
            if sparse_out is not None:
                sparse_out.append(
                    sparse_decay_rates_from_workspace(
                        entry.workspace,
                        spectral_density,
                        T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
        else:
            if operator_factory is None:
                raise ValueError(
                    "rates_at_flux_points needs entry.workspace or operator_factory."
                )
            states = list(entry.states)
            op = operator_factory(entry.system)
            sector_mats.append(
                sector_rate_matrix(
                    system=entry.system,
                    sectors=sectors,
                    states=states,
                    operator=op,
                    spectral_density=spectral_density,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
            )
            if sparse_out is not None:
                sparse_out.append(
                    compute_all_decay_rates(
                        system=entry.system,
                        states=states,
                        operator=op,
                        spectral_density=spectral_density,
                        T=T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
    return sector_mats, sparse_out
