# multiloop_fluxonium/system.py

from __future__ import annotations

"""System-level overlaps and spectral accessors for fluxoid sectors."""

import numpy as np

from .two_loop_fluxoid_model import (
    FluxoidModelParams,
    FluxoidSector,
    TwoLoopFluxoidModel,
    FluxoidSectorManager,
)
from ztpcraft.bosonic.oscillator_integrals.fock_states import OscillatorFockState
from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator

from ztpcraft.bosonic.oscillator_integrals.oscillator_operators import (
    OperatorMatrixBuilder,
)

from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class TwoLoopFluxoidSystem:
    """Facade combining sector eigensystems and inter-sector overlaps."""
    # Shared across all TwoLoopFluxoidSystem instances. This lets flux sweeps
    # reuse expensive Fock overlap matrices when oscillator geometry is the same.
    _GLOBAL_S_CACHE: dict[tuple[tuple[float, ...], tuple[float, ...], int], ComplexArray] = {}

    def __init__(
        self,
        params: FluxoidModelParams,
        cutoff: int = 120,
        evals_count: int = 10,
    ):
        """Initialize system and underlying sector manager.

        Parameters
        ----------
        params:
            Circuit and fluxoid model parameters.
        cutoff:
            Fluxonium basis cutoff passed to sector diagonalization.
        evals_count:
            Number of eigenpairs retained per sector.
        """
        self.model = TwoLoopFluxoidModel(params)

        # hidden internals
        self._manager = FluxoidSectorManager(
            self.model,
            cutoff=cutoff,
            evals_count=evals_count,
        )

        self._builder: OperatorMatrixBuilder = OperatorMatrixBuilder()

        self._S_cache: dict[tuple[float, int], ComplexArray] = {}

    @staticmethod
    def _array_fingerprint(array: NDArray[np.float64], *, digits: int = 12) -> tuple[float, ...]:
        """Return deterministic rounded tuple fingerprint for cache keys."""
        rounded = np.round(np.asarray(array, dtype=np.float64).ravel(), decimals=digits)
        return tuple(float(x) for x in rounded)

    @classmethod
    def _oscillator_fingerprint(cls, osc: LocalHarmonicOscillator) -> tuple[float, ...]:
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
            # Fall back to directional pair key for non-identical oscillators.
            return (
                (*geom, *cls._oscillator_fingerprint(osc2), *cls._array_fingerprint(osc2.center - osc1.center)),
                (),
                cutoff,
            )
        return (geom, cls._array_fingerprint(osc2.center - osc1.center), cutoff)

    @classmethod
    def clear_global_overlap_cache(cls) -> None:
        """Clear the process-wide overlap cache used across flux sweeps."""
        cls._GLOBAL_S_CACHE.clear()

    def _get_S(self, sector1: FluxoidSector, sector2: FluxoidSector) -> ComplexArray:
        """Build/cache Fock-basis overlap matrix between sector oscillators.

        Parameters
        ----------
        sector1:
            First sector label.
        sector2:
            Second sector label.

        Returns
        -------
        ComplexArray
            Overlap matrix between oscillator Fock bases.
        """
        s1 = self._manager.get_sector_state(sector1)
        s2 = self._manager.get_sector_state(sector2)

        cutoff = s1.cutoff
        if cutoff != s2.cutoff:
            raise ValueError("Sector cutoffs must match for overlap calculation.")

        # Keep local hot cache (fast path inside one system instance)
        delta = float(s2.shift - s1.shift)
        local_key = (round(delta, 12), cutoff)
        cached_local = self._S_cache.get(local_key)
        if cached_local is not None:
            return cached_local

        # Cross-system global cache (reuses overlap matrices across sweep points)
        global_key = self._global_overlap_key(s1.osc, s2.osc, cutoff)
        cached_global = self._GLOBAL_S_CACHE.get(global_key)
        if cached_global is not None:
            self._S_cache[local_key] = cached_global
            return cached_global

        # build S once
        S = np.zeros((cutoff, cutoff), dtype=np.complex128)

        for n in range(cutoff):
            for m in range(cutoff):
                state_n = OscillatorFockState(s1.osc, (n,))
                state_m = OscillatorFockState(s2.osc, (m,))
                S[n, m] = self._builder.overlap_states(state_n, state_m)

        self._S_cache[local_key] = S
        self._GLOBAL_S_CACHE[global_key] = S
        return S

    # -------------------------
    # Core API
    # -------------------------

    def overlap_matrix(
        self,
        sector1: FluxoidSector,
        sector2: FluxoidSector,
    ) -> ComplexArray:
        """Return eigenbasis overlap matrix between two sectors.

        Parameters
        ----------
        sector1:
            Bra-sector label.
        sector2:
            Ket-sector label.

        Returns
        -------
        ComplexArray
            Matrix `V1^dagger S V2` in sector-eigenstate bases.
        """

        s1 = self._manager.get_sector_state(sector1)
        s2 = self._manager.get_sector_state(sector2)

        V1 = s1.evecs
        V2 = s2.evecs

        S = self._get_S(sector1, sector2)

        return V1.conj().T @ S @ V2

    # -------------------------
    # Convenience accessors
    # -------------------------

    def eigenvalues_with_offset(self, sector: FluxoidSector) -> FloatArray:
        """Return sector eigenvalues shifted by fluxoid energy offset.

        Parameters
        ----------
        sector:
            Fluxoid sector label.

        Returns
        -------
        FloatArray
            Sector eigenvalues including model offset.
        """
        return self._manager.get_sector_state(sector).evals + self.model.energy_offset(
            sector
        )

    def eigenvectors_without_shift(self, sector: FluxoidSector) -> ComplexArray:
        """Return raw sector eigenvectors from diagonalization.

        Parameters
        ----------
        sector:
            Fluxoid sector label.

        Returns
        -------
        ComplexArray
            Eigenvector matrix in the underlying sector basis.
        """
        return self._manager.get_sector_state(sector).evecs
    
    def oscillator_center_shift(self, sector: FluxoidSector) -> float:
        """Return harmonic-oscillator center shift used for overlaps.

        Parameters
        ----------
        sector:
            Fluxoid sector label.

        Returns
        -------
        float
            Oscillator center shift associated with the sector.
        """
        return self._manager.get_sector_state(sector).shift
