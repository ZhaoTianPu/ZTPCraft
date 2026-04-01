# multiloop_fluxonium/system.py

from __future__ import annotations

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
    def __init__(
        self,
        params: FluxoidModelParams,
        cutoff: int = 120,
        evals_count: int = 10,
    ):
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
        return self._manager.get_sector_state(sector).evals + self.model.energy_offset(
            sector
        )

    def eigenvectors_without_shift(self, sector: FluxoidSector) -> ComplexArray:
        return self._manager.get_sector_state(sector).evecs
    
    def oscillator_center_shift(self, sector: FluxoidSector) -> float:
        return self._manager.get_sector_state(sector).shift
