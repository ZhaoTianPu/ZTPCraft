from __future__ import annotations

"""Parameter sweeps for the tunneling-assisted (phase-slip) FGR rates.

Diagonalization, perturbative hybridization, and dressed-operator construction
are expensive. For a fixed external-flux configuration,
:func:`build_fgr_workspace` caches energies, the basis-transform matrix, and
each channel's dressed operator matrix so temperature and spectral-density
variations only re-evaluate ``S(omega)`` (and thermal weights for sector
aggregation).
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ztpcraft.decoherence.fgr import FrequencyUnit, compute_rate_matrix

from .tunneling_assisted_hybridization import (
    GlobalState,
    PerturbativeHybridizationInfo,
    build_perturbative_hybridization_info,
)
from .tunneling_assisted_model import (
    PhaseSlipSector,
    TunnelingAssistedModelParams,
)
from .tunneling_assisted_system import PhaseSlipSystem
from .tunneling_assisted_transition_rates import (
    NoiseChannel,
    build_dressed_operator_matrix,
    build_global_states,
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


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSlipFGRWorkspace:
    """Cached hybridization and per-channel dressed operators for cheap sweeps.

    Build with :func:`build_fgr_workspace` using the same ``system`` that you
    pass to sector-rate helpers (for thermal weights). Energies and dressed
    operator matrices are fixed; only bath correlations and temperature enter
    afterward.

    Attributes
    ----------
    system:
        Phase-slip system used to build the workspace (and required for
        thermal-weight evaluation in sector aggregation).
    hybridization:
        Cached :class:`PerturbativeHybridizationInfo` (states, energies,
        basis-transform matrix ``U``, and diagnostics).
    channels:
        Tuple of noise channels included in the workspace, in order.
    dressed_operator_matrices:
        Mapping from ``channel.name`` to ``U^dagger O U`` in the truncated basis.
    """

    system: PhaseSlipSystem
    hybridization: PerturbativeHybridizationInfo
    channels: tuple[NoiseChannel, ...]
    dressed_operator_matrices: dict[str, ComplexArray]

    @property
    def states(self) -> tuple[GlobalState, ...]:
        return tuple(self.hybridization.states)

    @property
    def state_list(self) -> list[GlobalState]:
        return list(self.hybridization.states)

    @property
    def energies(self) -> FloatArray:
        return self.hybridization.energies

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    def channel(self, name: str) -> NoiseChannel:
        """Return the cached :class:`NoiseChannel` matching ``name``."""
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(f"No channel named {name!r}; have {self.channel_names!r}.")

    def dressed_operator(self, channel_name: str) -> ComplexArray:
        """Return the cached dressed operator matrix for ``channel_name``."""
        if channel_name not in self.dressed_operator_matrices:
            raise KeyError(
                f"No dressed operator cached for channel {channel_name!r}; "
                f"have {tuple(self.dressed_operator_matrices)!r}."
            )
        return self.dressed_operator_matrices[channel_name]


def build_fgr_workspace(
    system: PhaseSlipSystem,
    states: Sequence[GlobalState],
    channels: Sequence[NoiseChannel],
    *,
    hybridization: PerturbativeHybridizationInfo | None = None,
    perturbation_ratio_threshold: float = 0.2,
    detuning_floor: float = 1e-12,
) -> PhaseSlipFGRWorkspace:
    """Diagonalization-free cache for repeated FGR evaluations.

    Parameters
    ----------
    system:
        Phase-slip system used to build the dressed operators.
    states:
        Truncated global-state basis (same order used everywhere downstream).
    channels:
        Noise channels whose dressed operator matrices should be cached.
    hybridization:
        Optional precomputed :class:`PerturbativeHybridizationInfo`. If
        omitted, a fresh hybridization is built from ``states`` using
        ``perturbation_ratio_threshold`` and ``detuning_floor``.
    perturbation_ratio_threshold, detuning_floor:
        Forwarded to :func:`build_perturbative_hybridization_info` when
        ``hybridization`` is None.

    Returns
    -------
    PhaseSlipFGRWorkspace
        Cached hybridization + dressed operator matrices.
    """
    if len(channels) == 0:
        raise ValueError("channels must be non-empty.")

    seen: set[str] = set()
    for channel in channels:
        if channel.name in seen:
            raise ValueError(f"Duplicate channel name: {channel.name!r}")
        seen.add(channel.name)

    state_list = list(states)
    if hybridization is None:
        hybridization = build_perturbative_hybridization_info(
            system=system,
            states=state_list,
            perturbation_ratio_threshold=perturbation_ratio_threshold,
            detuning_floor=detuning_floor,
        )
    elif list(hybridization.states) != state_list:
        raise ValueError(
            "Provided hybridization.states does not match the supplied states."
        )

    dressed_operator_matrices: dict[str, ComplexArray] = {}
    for channel in channels:
        dressed_operator_matrices[channel.name] = build_dressed_operator_matrix(
            system=system,
            hybridization=hybridization,
            operator=channel.operator,
        )

    return PhaseSlipFGRWorkspace(
        system=system,
        hybridization=hybridization,
        channels=tuple(channels),
        dressed_operator_matrices=dressed_operator_matrices,
    )


# ---------------------------------------------------------------------------
# Workspace-driven evaluation
# ---------------------------------------------------------------------------


def rate_matrix_from_workspace(
    workspace: PhaseSlipFGRWorkspace,
    channel_name: str,
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    spectral_density: Callable[..., float | complex] | None = None,
) -> FloatArray:
    """FGR rate matrix using cached energies and a cached dressed operator.

    Parameters
    ----------
    workspace:
        Precomputed FGR workspace.
    channel_name:
        Name of the channel whose dressed operator is reused.
    T:
        Temperature in Kelvin (or None for spectral densities that do not take
        a temperature argument).
    spectral_density:
        Optional override of the channel's spectral density. Useful for
        spectral-model sweeps that share the same operator.
    units, spectral_omega_units:
        Forwarded to :func:`compute_rate_matrix`.
    """
    channel = workspace.channel(channel_name)
    sd = channel.spectral_density if spectral_density is None else spectral_density
    return compute_rate_matrix(
        energies=workspace.energies,
        O_matrix=workspace.dressed_operator(channel_name),
        spectral_density=sd,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
    )


def total_rate_matrix_from_workspace(
    workspace: PhaseSlipFGRWorkspace,
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    spectral_density_overrides: dict[str, Callable[..., float | complex]] | None = None,
) -> FloatArray:
    """Sum of dense rate matrices across all channels in ``workspace``."""
    overrides = spectral_density_overrides or {}
    n = len(workspace.states)
    total = np.zeros((n, n), dtype=np.float64)
    for channel in workspace.channels:
        total += rate_matrix_from_workspace(
            workspace=workspace,
            channel_name=channel.name,
            T=T,
            units=units,
            spectral_omega_units=spectral_omega_units,
            spectral_density=overrides.get(channel.name),
        )
    return total


def state_to_state_transition_rates_from_workspace(
    workspace: PhaseSlipFGRWorkspace,
    channel_name: str,
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    spectral_density: Callable[..., float | complex] | None = None,
) -> dict[tuple[int, int], float]:
    """Sparse ``{(i, j): rate}`` map for one cached channel."""
    rate_matrix = rate_matrix_from_workspace(
        workspace=workspace,
        channel_name=channel_name,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
        spectral_density=spectral_density,
    )
    O = workspace.dressed_operator(channel_name)
    rates: dict[tuple[int, int], float] = {}
    mask = np.abs(O) >= _MATRIX_ELEMENT_CUTOFF
    np.fill_diagonal(mask, False)
    for i, j in np.argwhere(mask):
        value = float(rate_matrix[int(i), int(j)])
        if value > 0.0:
            rates[(int(i), int(j))] = value
    return rates


def total_state_to_state_transition_rates_from_workspace(
    workspace: PhaseSlipFGRWorkspace,
    T: float | None,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    spectral_density_overrides: dict[str, Callable[..., float | complex]] | None = None,
) -> dict[tuple[int, int], float]:
    """Channel-summed sparse ``{(i, j): rate}`` map.

    Off-diagonal entries with positive total rate are returned.
    """
    total = total_rate_matrix_from_workspace(
        workspace=workspace,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
        spectral_density_overrides=spectral_density_overrides,
    )
    rates: dict[tuple[int, int], float] = {}
    n = total.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if total[i, j] > 0.0:
                rates[(i, j)] = float(total[i, j])
    return rates


def sector_rate_matrix_from_workspace(
    workspace: PhaseSlipFGRWorkspace,
    sectors: list[PhaseSlipSector],
    channel_name: str,
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    spectral_density: Callable[..., float | complex] | None = None,
) -> FloatArray:
    """Sector-aggregated rates for one channel without rebuilding the operator."""
    if T <= 0.0:
        raise ValueError("Temperature T must be positive for sector aggregation.")
    rates = rate_matrix_from_workspace(
        workspace=workspace,
        channel_name=channel_name,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
        spectral_density=spectral_density,
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


def total_sector_rate_matrix_from_workspace(
    workspace: PhaseSlipFGRWorkspace,
    sectors: list[PhaseSlipSector],
    T: float,
    *,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    spectral_density_overrides: dict[str, Callable[..., float | complex]] | None = None,
) -> FloatArray:
    """Channel-summed sector-aggregated rate matrix."""
    if T <= 0.0:
        raise ValueError("Temperature T must be positive for sector aggregation.")
    total = total_rate_matrix_from_workspace(
        workspace=workspace,
        T=T,
        units=units,
        spectral_omega_units=spectral_omega_units,
        spectral_density_overrides=spectral_density_overrides,
    )
    thermal_cache = {
        sector: thermal_weights(workspace.system, sector, T) for sector in sectors
    }
    return sector_rate_matrix_fast(
        states=workspace.state_list,
        sectors=sectors,
        rates=total,
        weights=thermal_cache,
    )


# ---------------------------------------------------------------------------
# Sweeps over T and spectral density (fixed flux)
# ---------------------------------------------------------------------------


def sweep_temperatures(
    workspace: PhaseSlipFGRWorkspace,
    sectors: list[PhaseSlipSector],
    temperatures: Sequence[float],
    *,
    channel_name: str | None = None,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    include_sparse: bool = False,
    show_progress: bool = True,
) -> list[FloatArray] | tuple[list[FloatArray], list[dict[tuple[int, int], float]]]:
    """Sector rate matrix at each temperature; operator matrices built once.

    Parameters
    ----------
    workspace:
        Precomputed FGR workspace for one flux configuration.
    sectors:
        Row/column order for the sector matrix.
    temperatures:
        Kelvin values (each must be ``> 0`` for thermal weights).
    channel_name:
        If provided, evaluate only that channel; otherwise sum all channels.
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
        desc="Phase-slip FGR temperature sweep",
        show_progress=show_progress,
    ):
        if channel_name is None:
            gamma = total_sector_rate_matrix_from_workspace(
                workspace=workspace,
                sectors=sectors,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
        else:
            gamma = sector_rate_matrix_from_workspace(
                workspace=workspace,
                sectors=sectors,
                channel_name=channel_name,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
        sector_mats.append(gamma)
        if include_sparse:
            if channel_name is None:
                sparse_list.append(
                    total_state_to_state_transition_rates_from_workspace(
                        workspace=workspace,
                        T=T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
            else:
                sparse_list.append(
                    state_to_state_transition_rates_from_workspace(
                        workspace=workspace,
                        channel_name=channel_name,
                        T=T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
    if include_sparse:
        return sector_mats, sparse_list
    return sector_mats


def sweep_spectral_densities(
    workspace: PhaseSlipFGRWorkspace,
    sectors: list[PhaseSlipSector],
    spectral_models: Sequence[Callable[..., float | complex]],
    T: float,
    *,
    channel_name: str,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    include_sparse: bool = False,
    show_progress: bool = True,
) -> list[FloatArray] | tuple[list[FloatArray], list[dict[tuple[int, int], float]]]:
    """Evaluate several spectral densities for one channel at fixed ``T`` and flux.

    The dressed operator for ``channel_name`` is cached in ``workspace`` and
    is reused for every spectral model.

    Parameters
    ----------
    spectral_models:
        Sequence of ``S(omega)`` or ``S(omega, T)`` callables to evaluate.
    channel_name:
        Channel whose operator (and ordering) is used. Each model in
        ``spectral_models`` overrides ``channel.spectral_density``.
    show_progress:
        If True, show a ``tqdm`` bar over spectral models.
    """
    sector_mats: list[FloatArray] = []
    sparse_list: list[dict[tuple[int, int], float]] = []
    for S in _maybe_progress(
        list(spectral_models),
        desc="Phase-slip FGR spectral-density sweep",
        show_progress=show_progress,
    ):
        gamma = sector_rate_matrix_from_workspace(
            workspace=workspace,
            sectors=sectors,
            channel_name=channel_name,
            T=T,
            units=units,
            spectral_omega_units=spectral_omega_units,
            spectral_density=S,
        )
        sector_mats.append(gamma)
        if include_sparse:
            sparse_list.append(
                state_to_state_transition_rates_from_workspace(
                    workspace=workspace,
                    channel_name=channel_name,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                    spectral_density=S,
                )
            )
    if include_sparse:
        return sector_mats, sparse_list
    return sector_mats


# ---------------------------------------------------------------------------
# External-flux sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSlipFluxSweepResult:
    """One external-flux sample with system, basis, hybridization, and workspace."""

    label: object
    params: TunnelingAssistedModelParams
    system: PhaseSlipSystem
    states: tuple[GlobalState, ...]
    hybridization: PerturbativeHybridizationInfo | None = None
    workspace: PhaseSlipFGRWorkspace | None = None


def _default_truncate(states: list[GlobalState]) -> list[GlobalState]:
    return states


def sweep_external_flux_transition_setup(
    *,
    sectors: Iterable[PhaseSlipSector],
    flux_labels_and_params: Sequence[tuple[object, TunnelingAssistedModelParams]],
    channels_factory: Callable[[PhaseSlipSystem], Sequence[NoiseChannel]],
    cutoff: int = 120,
    evals_count: int = 10,
    perturbation_ratio_threshold: float = 0.2,
    detuning_floor: float = 1e-12,
    truncate_states: Callable[[list[GlobalState]], list[GlobalState]] | None = None,
    build_workspace: bool = True,
    show_progress: bool = True,
) -> list[PhaseSlipFluxSweepResult]:
    """Build a fresh :class:`PhaseSlipSystem` per flux point.

    Each flux point performs full diagonalization + perturbative hybridization,
    and (by default) caches the dressed operator matrices for every channel.

    Parameters
    ----------
    sectors:
        Phase-slip sectors included in the global Hilbert space.
    flux_labels_and_params:
        ``(label, TunnelingAssistedModelParams)`` per sample. ``label`` is
        stored on the result (e.g. a ``phi_cm`` float or an integer index).
    channels_factory:
        Builds a sequence of :class:`NoiseChannel` for each new ``system``
        (e.g. ``lambda sys: [NoiseChannel("dielectric", "n_operator", S_diel),
        NoiseChannel("flux", "phi_operator", S_flux)]``). Channels may close
        over the system's params (e.g. ``EC``).
    cutoff, evals_count:
        Forwarded to :class:`PhaseSlipSystem`.
    perturbation_ratio_threshold, detuning_floor:
        Forwarded to :func:`build_perturbative_hybridization_info`.
    truncate_states:
        Optional callable applied to the full ``build_global_states`` output
        before hybridization (e.g. ``lambda states: [s for s in states if s.level < 4]``).
    build_workspace:
        If True (default), precompute :class:`PhaseSlipFGRWorkspace` for cheap
        T / spectral sweeps at each flux point.
    show_progress:
        If True (default), show a ``tqdm`` bar over flux points.

    Returns
    -------
    list[PhaseSlipFluxSweepResult]
        One entry per flux point.
    """
    sector_list = list(sectors)
    truncate = truncate_states or _default_truncate
    results: list[PhaseSlipFluxSweepResult] = []
    for label, params in _maybe_progress(
        list(flux_labels_and_params),
        desc="Phase-slip flux sweep (setup)",
        show_progress=show_progress,
    ):
        system = PhaseSlipSystem(params, cutoff=cutoff, evals_count=evals_count)
        full_states = build_global_states(sector_list, system)
        states = truncate(full_states)

        hybridization: PerturbativeHybridizationInfo | None = None
        workspace: PhaseSlipFGRWorkspace | None = None
        if build_workspace:
            channels = list(channels_factory(system))
            hybridization = build_perturbative_hybridization_info(
                system=system,
                states=states,
                perturbation_ratio_threshold=perturbation_ratio_threshold,
                detuning_floor=detuning_floor,
            )
            workspace = build_fgr_workspace(
                system=system,
                states=states,
                channels=channels,
                hybridization=hybridization,
            )
        results.append(
            PhaseSlipFluxSweepResult(
                label=label,
                params=params,
                system=system,
                states=tuple(states),
                hybridization=hybridization,
                workspace=workspace,
            )
        )
    return results


def jump_rates_flux_sweep(
    flux_results: Sequence[PhaseSlipFluxSweepResult],
    sectors: list[PhaseSlipSector],
    T: float,
    *,
    channel_name: str | None = None,
    channels_factory: Callable[[PhaseSlipSystem], Sequence[NoiseChannel]] | None = None,
    perturbation_ratio_threshold: float = 0.2,
    detuning_floor: float = 1e-12,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    save_state_to_state_transition_rates: bool = True,
    show_progress: bool = True,
) -> tuple[list[FloatArray], list[dict[tuple[int, int], float]] | None]:
    """Compute sector-to-sector (and optionally state-to-state) rates per flux point.

    Sector-to-sector rates assume the system thermalizes within each source
    sector before each jump (Boltzmann weighting over bare levels).

    If a result was created with ``build_workspace=True``, the cached workspace
    is reused. Otherwise ``channels_factory`` must be provided and a workspace
    is rebuilt on the fly for that flux point (still no extra work versus a
    manual loop).

    Parameters
    ----------
    flux_results:
        List of :class:`PhaseSlipFluxSweepResult` objects.
    sectors:
        Sector ordering for the output rate matrices.
    T:
        Temperature in Kelvin.
    channel_name:
        If provided, evaluate only that channel; otherwise sum all channels in
        the workspace.
    channels_factory:
        Required when any sweep entry was built with ``build_workspace=False``.
    perturbation_ratio_threshold, detuning_floor:
        Forwarded to hybridization construction when rebuilding workspaces.
    units, spectral_omega_units:
        Forwarded to FGR rate evaluation.
    save_state_to_state_transition_rates:
        If True, also return sparse ``{(i, j): rate}`` dictionaries.
    show_progress:
        If True (default), show a ``tqdm`` bar over flux points.

    Returns
    -------
    tuple[list[FloatArray], list[dict[tuple[int, int], float]] | None]
        ``(sector_to_sector_rates, state_to_state_rates)`` lists aligned with
        ``flux_results``.
    """
    sector_to_sector_rates: list[FloatArray] = []
    state_to_state_rates: list[dict[tuple[int, int], float]] | None = (
        [] if save_state_to_state_transition_rates else None
    )
    for entry in _maybe_progress(
        list(flux_results),
        desc="Phase-slip flux sweep (rates)",
        show_progress=show_progress,
    ):
        workspace = entry.workspace
        if workspace is None:
            if channels_factory is None:
                raise ValueError(
                    "jump_rates_flux_sweep needs entry.workspace or channels_factory."
                )
            channels = list(channels_factory(entry.system))
            workspace = build_fgr_workspace(
                system=entry.system,
                states=list(entry.states),
                channels=channels,
                hybridization=entry.hybridization,
                perturbation_ratio_threshold=perturbation_ratio_threshold,
                detuning_floor=detuning_floor,
            )

        if channel_name is None:
            gamma = total_sector_rate_matrix_from_workspace(
                workspace=workspace,
                sectors=sectors,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
        else:
            gamma = sector_rate_matrix_from_workspace(
                workspace=workspace,
                sectors=sectors,
                channel_name=channel_name,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
        sector_to_sector_rates.append(gamma)
        if state_to_state_rates is not None:
            if channel_name is None:
                state_to_state_rates.append(
                    total_state_to_state_transition_rates_from_workspace(
                        workspace=workspace,
                        T=T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
            else:
                state_to_state_rates.append(
                    state_to_state_transition_rates_from_workspace(
                        workspace=workspace,
                        channel_name=channel_name,
                        T=T,
                        units=units,
                        spectral_omega_units=spectral_omega_units,
                    )
                )
    return sector_to_sector_rates, state_to_state_rates


# ---------------------------------------------------------------------------
# Convenience: rebuild rates without the workspace cache (parity with Model A)
# ---------------------------------------------------------------------------


def jump_rates_flux_sweep_no_workspace(
    flux_results: Sequence[PhaseSlipFluxSweepResult],
    sectors: list[PhaseSlipSector],
    channels_factory: Callable[[PhaseSlipSystem], Sequence[NoiseChannel]],
    T: float,
    *,
    channel_name: str | None = None,
    perturbation_ratio_threshold: float = 0.2,
    detuning_floor: float = 1e-12,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    show_progress: bool = True,
) -> list[FloatArray]:
    """Compute sector rates per flux point without cached workspaces.

    Use when :func:`sweep_external_flux_transition_setup` was called with
    ``build_workspace=False``. This is equivalent to a manual loop calling
    :func:`sector_rate_matrix` (or summing across channels) at each flux point.
    """
    out: list[FloatArray] = []
    for entry in _maybe_progress(
        list(flux_results),
        desc="Phase-slip flux sweep (no-cache rates)",
        show_progress=show_progress,
    ):
        states = list(entry.states)
        hybridization = entry.hybridization or build_perturbative_hybridization_info(
            system=entry.system,
            states=states,
            perturbation_ratio_threshold=perturbation_ratio_threshold,
            detuning_floor=detuning_floor,
        )
        channels = list(channels_factory(entry.system))
        if channel_name is None:
            n_sectors = len(sectors)
            total = np.zeros((n_sectors, n_sectors), dtype=np.float64)
            for channel in channels:
                total += sector_rate_matrix(
                    system=entry.system,
                    hybridization=hybridization,
                    channel=channel,
                    sectors=sectors,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
            out.append(total)
        else:
            channel = next(c for c in channels if c.name == channel_name)
            out.append(
                sector_rate_matrix(
                    system=entry.system,
                    hybridization=hybridization,
                    channel=channel,
                    sectors=sectors,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
            )
    return out
