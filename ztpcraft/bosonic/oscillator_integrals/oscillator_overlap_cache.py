from collections.abc import Iterable, Sequence

import numpy as np

from .oscillators import LocalHarmonicOscillator
from .oscillator_overlap import OscillatorOverlapEngine

Occupation = tuple[int, ...]
PairKey = tuple[int, int]
OverlapValue = complex | tuple[complex, float | complex]
ValueCacheKey = tuple[PairKey, Occupation, Occupation, bool]


class OverlapManager:
    """
    Stateful manager for overlap computations.

    Features:
    - caches engines per oscillator pair
    - caches overlap values per (pair, occupations)
    - supports Hermitian symmetry reuse
    """

    def __init__(self) -> None:
        self._engines: dict[PairKey, OscillatorOverlapEngine] = {}
        self._value_cache: dict[ValueCacheKey, OverlapValue] = {}

    def _pair_key(
        self, osc1: LocalHarmonicOscillator, osc2: LocalHarmonicOscillator
    ) -> PairKey:
        return (id(osc1), id(osc2))

    def _get_engine(
        self, osc1: LocalHarmonicOscillator, osc2: LocalHarmonicOscillator
    ) -> OscillatorOverlapEngine:
        key = self._pair_key(osc1, osc2)
        if key not in self._engines:
            self._engines[key] = OscillatorOverlapEngine(osc1, osc2)
        return self._engines[key]

    def overlap(
        self,
        osc1: LocalHarmonicOscillator,
        osc2: LocalHarmonicOscillator,
        n1: Sequence[int] | Occupation,
        n2: Sequence[int] | Occupation,
        *,
        use_cache: bool = True,
        report_prefactor_as_exponent: bool = False,
    ) -> OverlapValue:
        n1 = tuple(n1)
        n2 = tuple(n2)

        pair_key = self._pair_key(osc1, osc2)
        cache_key = (pair_key, n1, n2, report_prefactor_as_exponent)

        if use_cache and cache_key in self._value_cache:
            return self._value_cache[cache_key]

        reverse_pair_key = (pair_key[1], pair_key[0])
        reverse_key = (reverse_pair_key, n2, n1, report_prefactor_as_exponent)

        if use_cache and reverse_key in self._value_cache:
            reverse_val = self._value_cache[reverse_key]
            if isinstance(reverse_val, tuple):
                val = (np.conjugate(reverse_val[0]), np.conjugate(reverse_val[1]))
            else:
                val = np.conjugate(reverse_val)
            self._value_cache[cache_key] = val
            return val

        engine = self._get_engine(osc1, osc2)
        val = engine.overlap(
            n1,
            n2,
            report_prefactor_as_exponent=report_prefactor_as_exponent,
        )

        if use_cache:
            self._value_cache[cache_key] = val

        return val

    def batch(
        self,
        osc1: LocalHarmonicOscillator,
        osc2: LocalHarmonicOscillator,
        pairs: Iterable[tuple[Sequence[int] | Occupation, Sequence[int] | Occupation]],
        *,
        report_prefactor_as_exponent: bool = False,
    ) -> list[OverlapValue]:
        engine = self._get_engine(osc1, osc2)
        out: list[OverlapValue] = []
        for n1, n2 in pairs:
            val = engine.overlap(
                n1, n2, report_prefactor_as_exponent=report_prefactor_as_exponent
            )
            out.append(val)
        return out

    def precompute(
        self,
        osc1: LocalHarmonicOscillator,
        osc2: LocalHarmonicOscillator,
        basis1: Iterable[Sequence[int] | Occupation],
        basis2: Iterable[Sequence[int] | Occupation],
        *,
        report_prefactor_as_exponent: bool = False,
    ) -> None:
        for n1 in basis1:
            for n2 in basis2:
                self.overlap(
                    osc1,
                    osc2,
                    n1,
                    n2,
                    report_prefactor_as_exponent=report_prefactor_as_exponent,
                )

    def clear_cache(self) -> None:
        self._value_cache.clear()

    def clear_engines(self) -> None:
        self._engines.clear()

    def stats(self) -> dict[str, int]:
        return {
            "num_engines": len(self._engines),
            "num_cached_values": len(self._value_cache),
        }


# TODO: support exponent mode (coeff, exponent)
# TODO: GPU batch evaluation
# TODO: disk persistence cache
# TODO: operator reuse (charge, cosine)


__all__ = ["OverlapManager"]
