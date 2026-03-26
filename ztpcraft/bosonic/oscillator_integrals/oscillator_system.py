from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .oscillators import LocalHarmonicOscillator
from .oscillator_overlap_cache import OverlapManager

Occupation = Sequence[int]


class OscillatorRegistry:
    """
    Stores oscillators (minima) and their associated basis states.
    """

    def __init__(self) -> None:
        self.oscillators: dict[str, LocalHarmonicOscillator] = {}
        self.bases: dict[str, list[Occupation]] = {}

    def add_minimum(self, name: str, oscillator: LocalHarmonicOscillator) -> None:
        self.oscillators[name] = oscillator

    def get_osc(self, name: str) -> LocalHarmonicOscillator:
        return self.oscillators[name]

    def set_basis(self, name: str, basis: list[Occupation]) -> None:
        self.bases[name] = basis

    def get_basis(self, name: str) -> list[Occupation]:
        return self.bases[name]


class OscillatorSystem:
    """
    High-level computation interface.
    """

    def __init__(self, registry: OscillatorRegistry) -> None:
        self.registry = registry
        self.manager = OverlapManager()

    def overlap(
        self, name1: str, name2: str, n1: Occupation, n2: Occupation
    ) -> complex:
        osc1 = self.registry.get_osc(name1)
        osc2 = self.registry.get_osc(name2)
        val = self.manager.overlap(osc1, osc2, n1, n2)
        assert not isinstance(val, tuple)
        return val

    def overlap_matrix(self, name1: str, name2: str) -> npt.NDArray[np.complex128]:
        osc1 = self.registry.get_osc(name1)
        osc2 = self.registry.get_osc(name2)

        basis1 = self.registry.get_basis(name1)
        basis2 = self.registry.get_basis(name2)

        mat: npt.NDArray[np.complex128] = np.zeros(
            (len(basis1), len(basis2)), dtype=np.complex128
        )

        for i, n1 in enumerate(basis1):
            for j, n2 in enumerate(basis2):
                val = self.manager.overlap(osc1, osc2, n1, n2)
                assert not isinstance(val, tuple)
                mat[i, j] = val

        return mat

    def precompute(self, name1: str, name2: str) -> None:
        osc1 = self.registry.get_osc(name1)
        osc2 = self.registry.get_osc(name2)

        basis1 = self.registry.get_basis(name1)
        basis2 = self.registry.get_basis(name2)

        for n1 in basis1:
            for n2 in basis2:
                self.manager.overlap(osc1, osc2, n1, n2)

    def stats(self) -> dict[str, int]:
        return self.manager.stats()


def sparse_to_full(
    indices: Sequence[int], values: Sequence[int], num_modes: int
) -> tuple[int, ...]:
    arr = [0] * num_modes
    for i, v in zip(indices, values):
        arr[i] = v
    return tuple(arr)


__all__ = ["Occupation", "OscillatorRegistry", "OscillatorSystem", "sparse_to_full"]
