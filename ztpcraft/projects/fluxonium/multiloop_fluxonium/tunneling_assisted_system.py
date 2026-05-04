from __future__ import annotations

"""System-level overlaps, tunneling matrix elements, and per-sector operator
accessors for the phase-slip reduced model."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ztpcraft.bosonic.oscillator_integrals.fock_states import OscillatorFockState
from ztpcraft.bosonic.oscillator_integrals.oscillator_operators import (
    OperatorMatrixBuilder,
)
from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator

from .tunneling_assisted_model import (
    PhaseSlipModel,
    PhaseSlipSector,
    PhaseSlipSectorManager,
    PhaseSlipSectorState,
    TunnelingAssistedModelParams,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# Local type alias for operator selectors. The canonical definition lives in
# ``tunneling_assisted_transition_rates``; we replicate it here to keep this
# module free of transition-rate imports.
OperatorSelector = str | Callable[[Any], NDArray[Any]]


class PhaseSlipSystem:
    """Facade combining per-sector eigensystems, inter-sector overlaps, and
    E_S-weighted tunneling matrix elements for the phase-slip reduced model.
    """

    # Shared across all PhaseSlipSystem instances so flux sweeps reuse expensive
    # Fock overlap matrices when oscillator geometry is the same.
    _GLOBAL_S_CACHE: dict[
        tuple[tuple[float, ...], tuple[float, ...], int], ComplexArray
    ] = {}

    def __init__(
        self,
        params: TunnelingAssistedModelParams,
        cutoff: int = 120,
        evals_count: int = 10,
    ):
        """Initialize system and underlying sector manager.

        Parameters
        ----------
        params:
            Phase-slip model parameters (circuit + external flux + ``E_S``).
        cutoff:
            Fluxonium basis cutoff passed to sector diagonalization.
        evals_count:
            Number of eigenpairs retained per sector.
        """
        self.model = PhaseSlipModel(params)
        self._manager = PhaseSlipSectorManager(
            self.model,
            cutoff=cutoff,
            evals_count=evals_count,
        )

        self._builder: OperatorMatrixBuilder = OperatorMatrixBuilder()

        self._S_cache: dict[tuple[float, int], ComplexArray] = {}
        self._state_overlap_cache: dict[tuple[int, int], ComplexArray] = {}
        self._operator_cache: dict[tuple[int, str], ComplexArray] = {}

    # ------------------------------------------------------------------
    # Global Fock-overlap cache utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _array_fingerprint(
        array: NDArray[np.float64], *, digits: int = 12
    ) -> tuple[float, ...]:
        """Return deterministic rounded tuple fingerprint for cache keys."""
        rounded = np.round(np.asarray(array, dtype=np.float64).ravel(), decimals=digits)
        return tuple(float(x) for x in rounded)

    @classmethod
    def _oscillator_fingerprint(
        cls, osc: LocalHarmonicOscillator
    ) -> tuple[float, ...]:
        """Fingerprint oscillator geometry excluding center shift."""
        return (
            *cls._array_fingerprint(osc.transform_xi_theta),
            *cls._array_fingerprint(osc.transform_qn),
            *cls._array_fingerprint(osc.frequencies),
        )

    @classmethod
    def _global_overlap_key(
        cls,
        osc1: LocalHarmonicOscillator,
        osc2: LocalHarmonicOscillator,
        cutoff: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...], int]:
        """Build global cache key from oscillator geometry and relative center."""
        geom = cls._oscillator_fingerprint(osc1)
        if geom != cls._oscillator_fingerprint(osc2):
            return (
                (
                    *geom,
                    *cls._oscillator_fingerprint(osc2),
                    *cls._array_fingerprint(osc2.center - osc1.center),
                ),
                (),
                cutoff,
            )
        return (geom, cls._array_fingerprint(osc2.center - osc1.center), cutoff)

    @classmethod
    def clear_global_overlap_cache(cls) -> None:
        """Clear the process-wide overlap cache used across flux sweeps."""
        cls._GLOBAL_S_CACHE.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_sector_state(self, sector: PhaseSlipSector) -> PhaseSlipSectorState:
        """Return the cached eigensystem/oscillator bundle for ``sector``."""
        return self._manager.get_sector_state(sector)

    def eigenvalues_with_offset(self, sector: PhaseSlipSector) -> FloatArray:
        """Return sector eigenvalues shifted by the phase-slip energy offset."""
        return self.get_sector_state(sector).evals + self.energy_offset(sector)

    def eigenvectors_without_shift(self, sector: PhaseSlipSector) -> ComplexArray:
        """Return raw sector eigenvectors from diagonalization."""
        return self.get_sector_state(sector).evecs

    def oscillator_center_shift(self, sector: PhaseSlipSector) -> float:
        """Return harmonic-oscillator center shift used for overlaps."""
        return self.get_sector_state(sector).shift

    def energy_offset(self, sector: PhaseSlipSector) -> float:
        """Delegate to ``self.model.energy_offset`` (overridable in tests)."""
        return self.model.energy_offset(sector)

    # ------------------------------------------------------------------
    # Fock and eigenbasis overlap matrices
    # ------------------------------------------------------------------

    def _get_S(
        self, sector1: PhaseSlipSector, sector2: PhaseSlipSector
    ) -> ComplexArray:
        """Build or retrieve the Fock-basis overlap matrix between sector oscillators.

        Parameters
        ----------
        sector1, sector2:
            Sector labels whose oscillator Fock bases are overlapped.

        Returns
        -------
        ComplexArray
            Overlap matrix ``S[n, m] = <n; sector1 | m; sector2>``.
        """
        s1 = self.get_sector_state(sector1)
        s2 = self.get_sector_state(sector2)

        cutoff = s1.cutoff
        if cutoff != s2.cutoff:
            raise ValueError("Sector cutoffs must match for overlap calculation.")

        delta = float(s2.shift - s1.shift)
        local_key = (round(delta, 12), cutoff)
        cached_local = self._S_cache.get(local_key)
        if cached_local is not None:
            return cached_local

        global_key = self._global_overlap_key(s1.osc, s2.osc, cutoff)
        cached_global = self._GLOBAL_S_CACHE.get(global_key)
        if cached_global is not None:
            self._S_cache[local_key] = cached_global
            return cached_global

        S = np.zeros((cutoff, cutoff), dtype=np.complex128)
        for n in range(cutoff):
            for m in range(cutoff):
                state_n = OscillatorFockState(s1.osc, (n,))
                state_m = OscillatorFockState(s2.osc, (m,))
                S[n, m] = self._builder.overlap_states(state_n, state_m)

        self._S_cache[local_key] = S
        self._GLOBAL_S_CACHE[global_key] = S
        return S

    def overlap_matrix(
        self, sector1: PhaseSlipSector, sector2: PhaseSlipSector
    ) -> ComplexArray:
        """Return eigenbasis overlap matrix ``V1^dagger S V2`` between two sectors."""
        key = (sector1.p, sector2.p)
        cached = self._state_overlap_cache.get(key)
        if cached is not None:
            return cached

        s1 = self.get_sector_state(sector1)
        s2 = self.get_sector_state(sector2)
        S = self._get_S(sector1, sector2)

        out = np.asarray(s1.evecs.conj().T @ S @ s2.evecs, dtype=np.complex128)
        self._state_overlap_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Tunneling matrix element (E_S-weighted)
    # ------------------------------------------------------------------

    def tunneling_matrix_element(
        self,
        source_sector: PhaseSlipSector,
        source_level: int,
        partner_sector: PhaseSlipSector,
        partner_level: int,
    ) -> complex:
        """Return ``E_S`` times the eigenbasis overlap for nearest-neighbor sectors.

        Zero unless ``|partner_sector.p - source_sector.p| == 1``.

        Parameters
        ----------
        source_sector, source_level:
            Source global-state label broken into its components.
        partner_sector, partner_level:
            Partner global-state label broken into its components.

        Returns
        -------
        complex
            Matrix element ``E_S * <partner | source>`` or ``0`` for non-neighbors.
        """
        dp = partner_sector.p - source_sector.p
        if abs(dp) != 1:
            return 0.0j
        overlap = self.overlap_matrix(partner_sector, source_sector)
        return complex(self.model.params.E_S) * complex(
            overlap[partner_level, source_level]
        )

    # ------------------------------------------------------------------
    # Per-sector operator in eigenbasis (uses the cached scqubits object)
    # ------------------------------------------------------------------

    def sector_operator_in_eigenbasis(
        self,
        sector: PhaseSlipSector,
        operator: OperatorSelector,
    ) -> ComplexArray:
        """Return a sector-local operator expressed in that sector's eigenbasis.

        Parameters
        ----------
        sector:
            Phase-slip sector whose fluxonium object provides the operator.
        operator:
            Either the name of an scqubits method (e.g. ``"n_operator"``) or a
            callable taking the fluxonium object and returning an operator matrix.

        Returns
        -------
        ComplexArray
            Operator matrix in the eigenbasis of ``sector``.
        """
        cache_key = (sector.p, repr(operator))
        cached = self._operator_cache.get(cache_key)
        if cached is not None:
            return cached

        state = self.get_sector_state(sector)
        if isinstance(operator, str):
            method = getattr(state.fluxonium, operator)
            if not callable(method):
                raise TypeError(f"Fluxonium attribute {operator!r} is not callable.")
            operator_matrix = method(energy_esys=(state.evals, state.evecs))
        else:
            operator_matrix = np.asarray(operator(state.fluxonium))

        op_array = np.asarray(operator_matrix, dtype=np.complex128)
        self._operator_cache[cache_key] = op_array
        return op_array
