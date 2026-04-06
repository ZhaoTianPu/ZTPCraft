from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
import scqubits as scq  # type: ignore
from scipy.constants import h, k as k_B

from ztpcraft.bosonic.oscillator_integrals.fock_states import OscillatorFockState
from ztpcraft.bosonic.oscillator_integrals.normal_modes import (
    diagonalize_quadratic_hamiltonian,
)
from ztpcraft.bosonic.oscillator_integrals.oscillator_operators import (
    OperatorMatrixBuilder,
)
from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator
from ztpcraft.decoherence.fgr import FrequencyUnit
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_transition_rates import (
    compute_rate_matrix,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_MATRIX_ELEMENT_CUTOFF = 1e-14


@dataclass(frozen=True)
class TunnelingAssistedModelParams:
    """Physical parameters for Model-B (tunneling-assisted incoherent jumps)."""
    EL_a: float
    EL_b: float
    EJ: float
    EC: float
    phi_ext_a: float
    phi_ext_b: float
    E_S: complex


@dataclass(frozen=True)
class PhaseSlipSector:
    """Phase-slip sector labeled by integer P."""
    P: int


@dataclass(frozen=True)
class PhaseSlipStateLabel:
    """Bare state label within a phase-slip sector."""
    sector: PhaseSlipSector
    level: int


@dataclass
class SpectrumInAPhaseSlipSector:
    """Cached eigensystem and oscillator metadata for one sector P."""
    sector: PhaseSlipSector
    evals: FloatArray
    evecs: ComplexArray
    shift: float
    cutoff: int
    osc: LocalHarmonicOscillator
    fluxonium: Any


@dataclass(frozen=True)
class HybridizationComponent:
    """Single perturbative contribution from a neighboring-sector bare state."""
    source: PhaseSlipStateLabel
    partner: PhaseSlipStateLabel
    coupling: complex
    detuning: float
    amplitude: complex
    ratio: float
    perturbation_valid: bool
    reason: str | None = None


@dataclass
class PerturbativeHybridizationInfo:
    """Hybridization output in the truncated bare basis.

    `dressed_eigenvectors_in_bare_basis` is the basis transform matrix U whose
    columns are dressed states expanded in the chosen bare basis.
    """
    states: list[PhaseSlipStateLabel]
    energies: FloatArray
    dressed_eigenvectors_in_bare_basis: ComplexArray
    contributions_by_source: dict[int, list[HybridizationComponent]]
    invalid_contributions: list[HybridizationComponent]


@dataclass(frozen=True)
class NoiseChannel:
    """Noise channel definition: operator + spectral density + user-facing name."""
    name: str
    operator: OperatorSelector
    spectral_density: Callable[..., float | complex]


OperatorSelector = str | Callable[[Any], NDArray[Any]]


class TunnelingAssistedIncoherentJumps:
    """Model-B perturbative tunneling-assisted incoherent jumps.

    Typical workflow:
    1) Build/choose a truncated basis and call `build_perturbative_hybridization_info`.
    2) Define one or more `NoiseChannel`s.
    3) Compute rates with `compute_multi_channel_decay_rates`.
    """

    def __init__(
        self,
        params: TunnelingAssistedModelParams,
        p_values: Sequence[int],
        cutoff: int = 120,
        evals_count: int = 10,
        perturbation_ratio_threshold: float = 0.2,
        detuning_floor: float = 1e-12,
    ):
        if len(p_values) == 0:
            raise ValueError("p_values must be non-empty.")
        if evals_count <= 0:
            raise ValueError("evals_count must be positive.")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if perturbation_ratio_threshold <= 0.0:
            raise ValueError("perturbation_ratio_threshold must be positive.")
        if detuning_floor <= 0.0:
            raise ValueError("detuning_floor must be positive.")

        self.params = params
        self.p_values = sorted(set(int(p) for p in p_values))
        self.cutoff = cutoff
        self.evals_count = evals_count
        self.perturbation_ratio_threshold = float(perturbation_ratio_threshold)
        self.detuning_floor = float(detuning_floor)

        self._builder = OperatorMatrixBuilder()
        self._sector_cache: dict[int, SpectrumInAPhaseSlipSector] = {}
        self._oscillator_overlap_matrix_cache: dict[tuple[float, int], ComplexArray] = (
            {}
        )
        self._state_overlap_matrix_cache: dict[tuple[int, int], ComplexArray] = {}
        self._operator_cache: dict[tuple[int, str], ComplexArray] = {}

    def inductive_fractions(self) -> tuple[float, float, float]:
        EL_sum = self.params.EL_a + self.params.EL_b
        return EL_sum, self.params.EL_a / EL_sum, self.params.EL_b / EL_sum

    def phi_cm_diff(self) -> tuple[float, float]:
        return (
            self.params.phi_ext_a + self.params.phi_ext_b,
            self.params.phi_ext_a - self.params.phi_ext_b,
        )

    def effective_flux(self, sector: PhaseSlipSector) -> float:
        _, f_a, f_b = self.inductive_fractions()
        phi_cm, phi_diff = self.phi_cm_diff()
        return (
            -2.0 * np.pi * f_a * sector.P + 0.5 * (f_b - f_a) * phi_cm + 0.5 * phi_diff
        )

    def energy_offset(self, sector: PhaseSlipSector) -> float:
        EL_sum, f_a, f_b = self.inductive_fractions()
        phi_cm, _ = self.phi_cm_diff()
        return 0.5 * EL_sum * f_a * f_b * (phi_cm + 2.0 * np.pi * sector.P) ** 2

    def coordinate_shift(self, sector: PhaseSlipSector) -> float:
        # For model B overlap construction, flux allocation is fully on the inductor.
        return self.effective_flux(sector)

    def _build_spectrum_in_a_sector(
        self, sector: PhaseSlipSector
    ) -> SpectrumInAPhaseSlipSector:
        EL_sum, _, _ = self.inductive_fractions()
        phi_eff = self.effective_flux(sector)
        flux = phi_eff / (2.0 * np.pi)
        fluxonium = scq.Fluxonium(  # type: ignore
            EJ=self.params.EJ,
            EC=self.params.EC,
            EL=EL_sum,
            flux=flux,
            cutoff=self.cutoff,
        )
        evals_raw, evecs_raw = cast(
            tuple[NDArray[Any], NDArray[Any]],
            fluxonium.eigensys(  # pyright: ignore[reportUnknownMemberType]
                evals_count=self.evals_count
            ),
        )
        evals = np.asarray(evals_raw, dtype=np.float64)
        evecs = np.asarray(evecs_raw, dtype=np.complex128)

        normal_modes = diagonalize_quadratic_hamiltonian(
            np.array([[self.params.EC]]), np.array([[EL_sum]])
        )
        shift = self.coordinate_shift(sector)
        osc = LocalHarmonicOscillator(
            transform_xi_theta=normal_modes.transform_xi_theta,
            transform_qn=normal_modes.transform_qn,
            frequencies=normal_modes.frequencies,
            center=np.array([shift]),
        )

        return SpectrumInAPhaseSlipSector(
            sector=sector,
            evals=evals,
            evecs=evecs,
            shift=shift,
            cutoff=self.cutoff,
            osc=osc,
            fluxonium=fluxonium,
        )

    def get_bare_spectrum_in_a_sector(
        self, sector: PhaseSlipSector
    ) -> SpectrumInAPhaseSlipSector:
        cached = self._sector_cache.get(sector.P)
        if cached is None:
            cached = self._build_spectrum_in_a_sector(sector)
            self._sector_cache[sector.P] = cached
        return cached

    def build_states(self) -> list[PhaseSlipStateLabel]:
        states: list[PhaseSlipStateLabel] = []
        for p in self.p_values:
            sector = PhaseSlipSector(P=p)
            evals = self.get_bare_spectrum_in_a_sector(sector).evals
            for level in range(len(evals)):
                states.append(PhaseSlipStateLabel(sector=sector, level=level))
        return states

    def build_energy_array(self, states: list[PhaseSlipStateLabel]) -> FloatArray:
        energies = np.empty(len(states), dtype=np.float64)
        for idx, state in enumerate(states):
            sector_state = self.get_bare_spectrum_in_a_sector(state.sector)
            energies[idx] = float(sector_state.evals[state.level]) + self.energy_offset(
                state.sector
            )
        return energies

    def _get_oscillator_overlap_matrix(
        self, sector1: PhaseSlipSector, sector2: PhaseSlipSector
    ) -> ComplexArray:
        sector_1_bare_spectrum = self.get_bare_spectrum_in_a_sector(sector1)
        sector_2_bare_spectrum = self.get_bare_spectrum_in_a_sector(sector2)
        delta = float(sector_2_bare_spectrum.shift - sector_1_bare_spectrum.shift)
        key = (round(delta, 12), sector_1_bare_spectrum.cutoff)
        cached = self._oscillator_overlap_matrix_cache.get(key)
        if cached is not None:
            return cached

        oscillator_overlap_matrix = np.zeros(
            (sector_1_bare_spectrum.cutoff, sector_2_bare_spectrum.cutoff),
            dtype=np.complex128,
        )
        for n in range(sector_1_bare_spectrum.cutoff):
            for m in range(sector_2_bare_spectrum.cutoff):
                state_n = OscillatorFockState(sector_1_bare_spectrum.osc, (n,))
                state_m = OscillatorFockState(sector_2_bare_spectrum.osc, (m,))
                oscillator_overlap_matrix[n, m] = self._builder.overlap_states(
                    state_n, state_m
                )

        self._oscillator_overlap_matrix_cache[key] = oscillator_overlap_matrix
        return oscillator_overlap_matrix

    def get_state_overlap_matrix(
        self, sector1: PhaseSlipSector, sector2: PhaseSlipSector
    ) -> ComplexArray:
        key = (sector1.P, sector2.P)
        cached = self._state_overlap_matrix_cache.get(key)
        if cached is not None:
            return cached

        sector_1_bare_spectrum = self.get_bare_spectrum_in_a_sector(sector1)
        sector_2_bare_spectrum = self.get_bare_spectrum_in_a_sector(sector2)
        oscillator_overlap_matrix = self._get_oscillator_overlap_matrix(
            sector1, sector2
        )
        state_overlap_matrix = (
            sector_1_bare_spectrum.evecs.conj().T
            @ oscillator_overlap_matrix
            @ sector_2_bare_spectrum.evecs
        )
        state_overlap_matrix = np.asarray(state_overlap_matrix, dtype=np.complex128)
        self._state_overlap_matrix_cache[key] = state_overlap_matrix
        return state_overlap_matrix

    def tunneling_matrix_element(
        self, source: PhaseSlipStateLabel, partner: PhaseSlipStateLabel
    ) -> complex:
        dP = partner.sector.P - source.sector.P
        if abs(dP) != 1:
            return 0.0j
        state_overlap_matrix = self.get_state_overlap_matrix(
            partner.sector, source.sector
        )
        return complex(self.params.E_S) * complex(
            state_overlap_matrix[partner.level, source.level]
        )

    @staticmethod
    def _state_structure(
        states: list[PhaseSlipStateLabel],
    ) -> tuple[dict[int, NDArray[np.int64]], dict[int, NDArray[np.int64]]]:
        indices: dict[int, list[int]] = {}
        levels: dict[int, list[int]] = {}
        for idx, state in enumerate(states):
            p = state.sector.P
            indices.setdefault(p, []).append(idx)
            levels.setdefault(p, []).append(state.level)
        return (
            {p: np.asarray(v, dtype=np.int64) for p, v in indices.items()},
            {p: np.asarray(v, dtype=np.int64) for p, v in levels.items()},
        )

    def _single_sector_operator_in_bare_eigenbasis(
        self,
        sector: PhaseSlipSector,
        operator: OperatorSelector,
    ) -> ComplexArray:
        cache_key = (sector.P, repr(operator))
        cached = self._operator_cache.get(cache_key)
        if cached is not None:
            return cached

        sector_state = self.get_bare_spectrum_in_a_sector(sector)
        if isinstance(operator, str):
            method = getattr(sector_state.fluxonium, operator)
            if not callable(method):
                raise TypeError(f"Fluxonium attribute {operator!r} is not callable.")
            operator_matrix = method(
                energy_esys=(sector_state.evals, sector_state.evecs)
            )
        else:
            operator_matrix = np.asarray(operator(sector_state.fluxonium))

        self._operator_cache[cache_key] = operator_matrix  # type: ignore[assignment]
        return operator_matrix  # type: ignore[return-value]

    def build_bare_operator_matrix(
        self,
        states: list[PhaseSlipStateLabel],
        operator: OperatorSelector,
    ) -> ComplexArray:
        n_states = len(states)
        matrix = np.zeros((n_states, n_states), dtype=np.complex128)
        sector_indices, sector_levels = self._state_structure(states)
        for p, idx in sector_indices.items():
            sector = PhaseSlipSector(P=p)
            op_sector = self._single_sector_operator_in_bare_eigenbasis(
                sector, operator
            )
            levels = sector_levels[p]
            block = op_sector[np.ix_(levels, levels)]
            matrix[np.ix_(idx, idx)] = block
        return matrix

    @staticmethod
    def _neighbor_ps(p: int, allowed_p_values: set[int]) -> tuple[int, ...]:
        neighbors: list[int] = []
        if p - 1 in allowed_p_values:
            neighbors.append(p - 1)
        if p + 1 in allowed_p_values:
            neighbors.append(p + 1)
        return tuple(neighbors)

    def build_perturbative_hybridization_info(
        self, states: list[PhaseSlipStateLabel] | None = None
    ) -> PerturbativeHybridizationInfo:
        """Build first-order hybridization in a truncated bare basis.

        If `states` is None, all states from `self.p_values` and `evals_count`
        are used. If provided, only those labels are used, which directly sets
        the truncation used for perturbative mixing.

        Parameters
        ----------
        states:
            Optional explicit truncated bare basis. Each entry is a
            `PhaseSlipStateLabel(sector=P, level=mu)`.

        Returns
        -------
        PerturbativeHybridizationInfo
            Hybridization data including energies, basis transform matrix, and
            channel-by-channel perturbative contribution diagnostics.
        """
        basis_states = self.build_states() if states is None else list(states)
        # n_states is the total number of states that we consider
        n_states = len(basis_states)
        energies = self.build_energy_array(basis_states)
        allowed_p_values = {state.sector.P for state in basis_states}

        state_to_index = {state: idx for idx, state in enumerate(basis_states)}
        # "mixing_matrix" is a basis transform U from bare -> dressed components.
        # U has identity on the source-sector component and dense neighbor-sector
        # blocks from perturbative amplitudes.
        dressed_eigenvectors_in_bare_basis = np.eye(n_states, dtype=np.complex128)
        hybridization_components_for_all_states: dict[
            int, list[HybridizationComponent]
        ] = {idx: [] for idx in range(n_states)}
        invalid_components: list[HybridizationComponent] = []

        for source_state_idx, source_state in enumerate(basis_states):
            neighbor_phase_slip_indices = self._neighbor_ps(
                source_state.sector.P, allowed_p_values
            )
            for neighbor_p in neighbor_phase_slip_indices:
                partner_state_sector = PhaseSlipSector(P=neighbor_p)
                # only for getting total number of levels in the partner sector
                # not for getting the actual energies
                partner_sector_bare_evals_without_offset = (
                    self.get_bare_spectrum_in_a_sector(partner_state_sector).evals
                )
                for partner_level_idx in range(
                    len(partner_sector_bare_evals_without_offset)
                ):
                    partner_bare_state_label = PhaseSlipStateLabel(
                        sector=partner_state_sector, level=partner_level_idx
                    )
                    partner_idx = state_to_index.get(partner_bare_state_label)
                    if partner_idx is None:
                        continue

                    coupling = self.tunneling_matrix_element(
                        source_state, partner_bare_state_label
                    )
                    detuning = float(energies[source_state_idx] - energies[partner_idx])

                    if abs(detuning) < self.detuning_floor:
                        amplitude = 0.0j
                        ratio = float("inf")
                        valid = False
                        reason = (
                            f"|detuning| < detuning_floor ({self.detuning_floor:.3e})"
                        )
                    else:
                        amplitude = coupling / detuning
                        ratio = abs(amplitude)
                        valid = bool(ratio <= self.perturbation_ratio_threshold)
                        reason = (
                            None if valid else "hybridization_ratio_exceeds_threshold"
                        )

                    hybridization_component = HybridizationComponent(
                        source=source_state,
                        partner=partner_bare_state_label,
                        coupling=complex(coupling),
                        detuning=detuning,
                        amplitude=complex(amplitude),
                        ratio=float(ratio),
                        perturbation_valid=valid,
                        reason=reason,
                    )
                    hybridization_components_for_all_states[source_state_idx].append(
                        hybridization_component
                    )
                    if not valid:
                        invalid_components.append(hybridization_component)

                    if abs(amplitude) > _MATRIX_ELEMENT_CUTOFF:
                        dressed_eigenvectors_in_bare_basis[
                            partner_idx, source_state_idx
                        ] += amplitude

        return PerturbativeHybridizationInfo(
            states=list(basis_states),
            energies=energies,
            dressed_eigenvectors_in_bare_basis=dressed_eigenvectors_in_bare_basis,
            contributions_by_source=hybridization_components_for_all_states,
            invalid_contributions=invalid_components,
        )

    def build_dressed_operator_matrix(
        self,
        hybridization: PerturbativeHybridizationInfo,
        operator: OperatorSelector,
        bare_operator_matrix: ComplexArray | None = None,
    ) -> ComplexArray:
        """Return dressed operator U^dagger O U in the chosen truncated basis.

        Parameters
        ----------
        hybridization:
            Hybridization result from `build_perturbative_hybridization_info`.
        operator:
            Operator selector. Either an scqubits operator method name (e.g.
            `"n_operator"`, `"phi_operator"`) or a callable taking a fluxonium
            object and returning an operator matrix.
        bare_operator_matrix:
            Optional precomputed bare-basis operator matrix matching
            `hybridization.states`.

        Returns
        -------
        ComplexArray
            Dressed operator matrix in the same truncated basis ordering as
            `hybridization.states`.
        """
        bare_operator = (
            self.build_bare_operator_matrix(hybridization.states, operator)
            if bare_operator_matrix is None
            else bare_operator_matrix
        )
        U = hybridization.dressed_eigenvectors_in_bare_basis
        return np.asarray(U.conj().T @ bare_operator @ U, dtype=np.complex128)

    def compute_transition_rate_matrix_for_channel(
        self,
        hybridization: PerturbativeHybridizationInfo,
        channel: NoiseChannel,
        T: float | None,
        units: FrequencyUnit = "GHz",
        spectral_omega_units: Literal["input", "SI"] = "SI",
    ) -> FloatArray:
        """Compute full transition-rate matrix for one noise channel.

        Parameters
        ----------
        hybridization:
            Hybridization result defining basis and energies.
        channel:
            Noise channel containing operator and spectral density.
        T:
            Temperature in Kelvin. Pass `None` if spectral density is
            temperature-independent.
        units:
            Energy/frequency unit used by `hybridization.energies`.
        spectral_omega_units:
            Units expected by the spectral density function.

        Returns
        -------
        FloatArray
            Full rate matrix with shape `(n_states, n_states)`.
        """
        dressed_operator = self.build_dressed_operator_matrix(
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

    def _compute_multi_channel_rate_matrices(
        self,
        channels: Sequence[NoiseChannel],
        T: float | None,
        states: list[PhaseSlipStateLabel] | None = None,
        hybridization: PerturbativeHybridizationInfo | None = None,
        units: FrequencyUnit = "GHz",
        spectral_omega_units: Literal["input", "SI"] = "SI",
    ) -> tuple[
        PerturbativeHybridizationInfo, dict[str, FloatArray], FloatArray
    ]:
        """Internal helper returning per-channel and summed dense rate matrices.

        Parameters
        ----------
        channels:
            Sequence of noise channels to include.
        T:
            Temperature in Kelvin.
        states:
            Optional explicit truncated basis.
        hybridization:
            Optional precomputed hybridization. If provided, `states` is ignored.
        units:
            Energy/frequency unit used by state energies.
        spectral_omega_units:
            Units expected by spectral density callables.

        Returns
        -------
        tuple[PerturbativeHybridizationInfo, dict[str, FloatArray], FloatArray]
            `(hybridization, per_channel_rate_matrices, total_rate_matrix)`.
        """
        if len(channels) == 0:
            raise ValueError("channels must be non-empty.")

        seen: set[str] = set()
        for channel in channels:
            if channel.name in seen:
                raise ValueError(f"Duplicate channel name: {channel.name!r}")
            seen.add(channel.name)

        hybrid = (
            self.build_perturbative_hybridization_info(states=states)
            if hybridization is None
            else hybridization
        )
        total = np.zeros((len(hybrid.states), len(hybrid.states)), dtype=np.float64)
        per_channel: dict[str, FloatArray] = {}

        for channel in channels:
            rate_matrix = self.compute_transition_rate_matrix_for_channel(
                hybridization=hybrid,
                channel=channel,
                T=T,
                units=units,
                spectral_omega_units=spectral_omega_units,
            )
            per_channel[channel.name] = rate_matrix
            total += rate_matrix

        return hybrid, per_channel, total

    def compute_multi_channel_decay_rates(
        self,
        channels: Sequence[NoiseChannel],
        T: float | None,
        states: list[PhaseSlipStateLabel] | None = None,
        hybridization: PerturbativeHybridizationInfo | None = None,
        units: FrequencyUnit = "GHz",
        spectral_omega_units: Literal["input", "SI"] = "SI",
    ) -> tuple[dict[str, dict[tuple[int, int], float]], dict[tuple[int, int], float]]:
        """Compute sparse decay maps for multiple channels.

        Returns `(per_channel, total)` where each value is a dictionary keyed by
        `(i, j)` state-index pairs for positive off-diagonal rates only.

        Parameters
        ----------
        channels:
            Sequence of noise channels to include.
        T:
            Temperature in Kelvin.
        states:
            Optional explicit truncated basis used to build hybridization.
        hybridization:
            Optional precomputed hybridization. If provided, `states` is ignored.
        units:
            Energy/frequency unit used by state energies.
        spectral_omega_units:
            Units expected by spectral density callables.

        Returns
        -------
        tuple[dict[str, dict[tuple[int, int], float]], dict[tuple[int, int], float]]
            `(per_channel, total)` sparse transition dictionaries.
        """
        _, rate_matrix_by_channel, total_matrix = self._compute_multi_channel_rate_matrices(
            channels=channels,
            T=T,
            states=states,
            hybridization=hybridization,
            units=units,
            spectral_omega_units=spectral_omega_units,
        )
        per_channel_decay: dict[str, dict[tuple[int, int], float]] = {}
        for name, rate_matrix in rate_matrix_by_channel.items():
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

    def thermal_weights(self, sector: PhaseSlipSector, T: float) -> FloatArray:
        """Boltzmann weights within a fixed sector P.

        Parameters
        ----------
        sector:
            Phase-slip sector for thermal weighting.
        T:
            Temperature in Kelvin.

        Returns
        -------
        FloatArray
            Normalized Boltzmann weights over bare levels in `sector`.
        """
        if T <= 0.0:
            raise ValueError("Temperature T must be positive.")
        energies = np.asarray(
            self.get_bare_spectrum_in_a_sector(sector).evals, dtype=np.float64
        ) + self.energy_offset(sector)
        min_energy = float(np.min(energies))
        beta = h / (k_B * T)
        exponents = -beta * (energies - min_energy) * 1e9
        weights = np.exp(exponents)
        partition = float(np.sum(weights))
        if partition <= 0.0:
            raise ValueError("Thermal partition function is non-positive.")
        return np.asarray(weights / partition, dtype=np.float64)

    def p_to_p_rate(
        self,
        hybridization: PerturbativeHybridizationInfo,
        channel: NoiseChannel,
        p_from: int,
        p_to: int,
        T: float,
        units: FrequencyUnit = "GHz",
        spectral_omega_units: Literal["input", "SI"] = "SI",
    ) -> float:
        """Thermally averaged rate from sector `p_from` to `p_to` for one channel.

        Parameters
        ----------
        hybridization:
            Hybridization result defining basis and energies.
        channel:
            Noise channel used to compute transition rates.
        p_from:
            Initial phase-slip sector.
        p_to:
            Final phase-slip sector.
        T:
            Temperature in Kelvin.
        units:
            Energy/frequency unit used by state energies.
        spectral_omega_units:
            Units expected by the spectral density callable.

        Returns
        -------
        float
            Thermally averaged sector-to-sector transition rate.
        """
        rates = self.compute_transition_rate_matrix_for_channel(
            hybridization=hybridization,
            channel=channel,
            T=T,
            units=units,
            spectral_omega_units=spectral_omega_units,
        )
        states = hybridization.states
        indices_from = np.asarray(
            [i for i, s in enumerate(states) if s.sector.P == p_from], dtype=np.int64
        )
        indices_to = np.asarray(
            [i for i, s in enumerate(states) if s.sector.P == p_to], dtype=np.int64
        )
        if indices_from.size == 0 or indices_to.size == 0:
            return 0.0

        levels_from = np.asarray(
            [states[int(i)].level for i in indices_from], dtype=np.int64
        )
        weights = self.thermal_weights(PhaseSlipSector(P=p_from), T)[levels_from]
        block = rates[np.ix_(indices_from, indices_to)]
        return float(np.sum(weights[:, None] * block))

    def p_rate_matrix(
        self,
        hybridization: PerturbativeHybridizationInfo,
        channel: NoiseChannel,
        T: float,
        units: FrequencyUnit = "GHz",
        spectral_omega_units: Literal["input", "SI"] = "SI",
    ) -> FloatArray:
        """Sector-to-sector thermal rate matrix for one channel.

        Parameters
        ----------
        hybridization:
            Hybridization result defining basis and energies.
        channel:
            Noise channel used to compute rates.
        T:
            Temperature in Kelvin.
        units:
            Energy/frequency unit used by state energies.
        spectral_omega_units:
            Units expected by the spectral density callable.

        Returns
        -------
        FloatArray
            Matrix `R[a,b] = Gamma_{P_a -> P_b}` using thermal averaging within
            each source sector.
        """
        p_list = list(self.p_values)
        matrix = np.zeros((len(p_list), len(p_list)), dtype=np.float64)
        for i, p_from in enumerate(p_list):
            for j, p_to in enumerate(p_list):
                matrix[i, j] = self.p_to_p_rate(
                    hybridization=hybridization,
                    channel=channel,
                    p_from=p_from,
                    p_to=p_to,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
        return matrix

    def aggregated_neighbor_jump_rates(
        self,
        hybridization: PerturbativeHybridizationInfo,
        channel: NoiseChannel,
        T: float,
        units: FrequencyUnit = "GHz",
        spectral_omega_units: Literal["input", "SI"] = "SI",
    ) -> dict[int, dict[int, float]]:
        """Return aggregated nearest-neighbor (P->P±1) thermal rates.

        Parameters
        ----------
        hybridization:
            Hybridization result defining basis and energies.
        channel:
            Noise channel used to compute rates.
        T:
            Temperature in Kelvin.
        units:
            Energy/frequency unit used by state energies.
        spectral_omega_units:
            Units expected by the spectral density callable.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested mapping `{P_from: {P_to: rate}}` restricted to nearest-neighbor
            targets `P_to in {P_from-1, P_from+1}` present in the configured range.
        """
        out: dict[int, dict[int, float]] = {}
        for p in self.p_values:
            out[p] = {}
            for p_target in self._neighbor_ps(p, set(self.p_values)):
                out[p][p_target] = self.p_to_p_rate(
                    hybridization=hybridization,
                    channel=channel,
                    p_from=p,
                    p_to=p_target,
                    T=T,
                    units=units,
                    spectral_omega_units=spectral_omega_units,
                )
        return out


def ensure_explicit_p_values(p_values: Iterable[int]) -> list[int]:
    """Normalize user-provided P values (unique + sorted) and validate non-empty.

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
        raise ValueError("At least one explicit P value is required.")
    return out


__all__ = [
    "HybridizationComponent",
    "NoiseChannel",
    "OperatorSelector",
    "PerturbativeHybridizationInfo",
    "PhaseSlipSector",
    "SpectrumInAPhaseSlipSector",
    "PhaseSlipStateLabel",
    "TunnelingAssistedIncoherentJumps",
    "TunnelingAssistedModelParams",
    "ensure_explicit_p_values",
]
