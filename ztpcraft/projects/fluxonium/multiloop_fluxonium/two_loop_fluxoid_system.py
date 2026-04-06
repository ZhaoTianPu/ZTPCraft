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

from ztpcraft.bosonic.oscillator_integrals.oscillator_operators import (
    OperatorMatrixBuilder,
)

from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class TwoLoopFluxoidSystem:
    """Facade combining sector eigensystems and inter-sector overlaps."""

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

        # key depends only on relative shift
        delta = float(s2.shift - s1.shift)
        key = (round(delta, 12), cutoff)

        if key in self._S_cache:
            return self._S_cache[key]

        # build S once
        S = np.zeros((cutoff, cutoff), dtype=np.complex128)

        for n in range(cutoff):
            for m in range(cutoff):
                state_n = OscillatorFockState(s1.osc, (n,))
                state_m = OscillatorFockState(s2.osc, (m,))
                S[n, m] = self._builder.overlap_states(state_n, state_m)

        self._S_cache[key] = S
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
