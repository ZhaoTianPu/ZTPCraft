from __future__ import annotations

"""Parameter sweeps for two-loop fluxoid FGR rates.

Diagonalization and global operator-matrix assembly are expensive. For a fixed
external-flux configuration, :func:`build_fgr_workspace` caches energies and the
dense operator matrix so temperature and spectral-density variations only
re-evaluate ``S(ω)`` (and thermal weights for sector aggregation).
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

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
ComplexArray = NDArray[np.complex128]

_MATRIX_ELEMENT_CUTOFF = 1e-14

_T_seq = TypeVar("_T_seq")


def _maybe_progress(
    seq: Sequence[_T_seq],
    *,
    desc: str,
    show_progress: bool,
) -> Iterable[_T_seq]:
    """Wrap ``seq`` in ``tqdm`` when ``show_progress`` is True."""
    if not show_progress:
        return seq
    return tqdm(seq, desc=desc)


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


def state_to_state_transition_rates_from_workspace(
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
    show_progress: bool = True,
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
    show_progress:
        If True (default), show a ``tqdm`` bar over temperatures.

    Returns
    -------
    list[FloatArray]
        Sector matrices, or ``(sector_matrices, sparse_rates)`` if
        ``include_sparse`` is True.
    """
    sector_mats: list[FloatArray] = []
    sparse_list: list[dict[tuple[int, int], float]] = []
    for T in _maybe_progress(
        list(temperatures),
        desc="FGR temperature sweep",
        show_progress=show_progress,
    ):
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
                state_to_state_transition_rates_from_workspace(
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
    show_progress: bool = True,
) -> list[FloatArray] | tuple[list[FloatArray], list[dict[tuple[int, int], float]]]:
    """Evaluate several spectral-density callables at fixed temperature and flux.

    Use this when parameters enter only through ``S`` (e.g. amplitude, cutoff
    frequency) so Hamiltonian overlaps are reused.

    Parameters
    ----------
    show_progress:
        If True, show a ``tqdm`` bar over spectral models.
    """
    sector_mats: list[FloatArray] = []
    sparse_list: list[dict[tuple[int, int], float]] = []
    for S in _maybe_progress(
        list(spectral_models),
        desc="FGR spectral-density sweep",
        show_progress=show_progress,
    ):
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
                state_to_state_transition_rates_from_workspace(
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
    show_progress: bool = True,
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
    show_progress:
        If True, show a ``tqdm`` bar over flux points.

    Returns
    -------
    list[FluxoidFluxSweepResult]
        One entry per flux point.
    """
    results: list[FluxoidFluxSweepResult] = []
    sector_list = list(sectors)
    for label, params in _maybe_progress(
        list(flux_labels_and_params),
        desc="Fluxoid flux sweep (setup)",
        show_progress=show_progress,
    ):
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


def jump_rates_flux_sweep(
    flux_results: Sequence[FluxoidFluxSweepResult],
    sectors: list[FluxoidSector],
    spectral_density: Callable[..., float | complex],
    T: float,
    *,
    operator_factory: Callable[[TwoLoopFluxoidSystem], FluxoidOperator] | None = None,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    save_state_to_state_transition_rates: bool = True,
    show_progress: bool = True,
) -> tuple[list[FloatArray], list[dict[tuple[int, int], float]] | None]:
    """Compute sector-to-sector (and optionally state-to-state) transition rates for pre-built flux sweep results.

    When computing sector-to-sector transition rates, we assume that before each jump, the system thermalizes within
    each sector.

    If results were created with ``build_workspace=True``, uses the cached
    operator matrix. If ``workspace`` is missing, pass ``operator_factory`` to
    rebuild ``O`` once per flux point (still no extra work versus a manual loop).

    Parameters
    ----------
    flux_results:
        List of :class:`FluxoidFluxSweepResult` objects.
    sectors:
        List of :class:`FluxoidSector` objects.
    spectral_density:
        Spectral density callable.
    T:
        Temperature in Kelvin.
    operator_factory:
        Operator factory callable.
    units:
        Unit used by energies.
    spectral_omega_units:
        Units expected by spectral-density callable.
    save_state_to_state_transition_rates:
        If True, save state-to-state transition rates.
    show_progress:
        If True (default), show a ``tqdm`` bar over flux results.

    Returns
    -------
    tuple[list[FloatArray], list[dict[tuple[int, int], float]] | None]
        List of sector-to-sector transition rates, and optionally list of state-to-state transition rates.
    """
    sector_to_sector_transition_rates: list[FloatArray] = []
    state_to_state_transition_rates: list[dict[tuple[int, int], float]] | None = (
        [] if save_state_to_state_transition_rates else None
    )
    for sweep_entry in _maybe_progress(
        list(flux_results),
        desc="Fluxoid flux sweep (rates)",
        show_progress=show_progress,
    ):
        if sweep_entry.workspace is not None:
            gamma = sector_rate_matrix_from_workspace(
                sweep_entry.workspace,
                sectors,
                spectral_density,
                T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
            sector_to_sector_transition_rates.append(gamma)
            if state_to_state_transition_rates is not None:
                state_to_state_transition_rates.append(
                    state_to_state_transition_rates_from_workspace(
                        sweep_entry.workspace,
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
            states = list(sweep_entry.states)
            op = operator_factory(sweep_entry.system)
            sector_to_sector_transition_rates.append(
                sector_rate_matrix(
                    system=sweep_entry.system,
                    sectors=sectors,
                    states=states,
                    operator=op,
                    spectral_density=spectral_density,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
            )
            if state_to_state_transition_rates is not None:
                state_to_state_transition_rates.append(
                    compute_all_decay_rates(
                        system=sweep_entry.system,
                        states=states,
                        operator=op,
                        spectral_density=spectral_density,
                        T=T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
    return sector_to_sector_transition_rates, state_to_state_transition_rates
