from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .oscillator_overlap_cache import OverlapManager
from .oscillators import LocalHarmonicOscillator
from .fock_states import OscillatorFockState

Occupation = tuple[int, ...]


def shift_occupation(n: Occupation, index: int, delta: int) -> Occupation | None:
    n_new = list(n)
    n_new[index] += delta
    if n_new[index] < 0:
        return None
    return tuple(n_new)


def n_operator_element(
    osc_bra: LocalHarmonicOscillator,
    osc_ket: LocalHarmonicOscillator,
    n_bra: Occupation,
    n_ket: Occupation,
    charge_index: int,
    overlap_manager: OverlapManager,
) -> complex:
    n_bra = tuple(n_bra)
    n_ket = tuple(n_ket)
    q_to_n = osc_ket.q_to_n_matrix()

    val: complex = 0.0 + 0.0j
    for mu, freq in enumerate(osc_ket.frequencies):
        coeff = q_to_n[charge_index, mu]
        prefactor = 1j * np.sqrt(freq / 2.0)
        s_mu = n_ket[mu]

        n_plus = shift_occupation(n_ket, mu, +1)
        term_plus: complex = 0.0 + 0.0j
        if n_plus is not None:
            overlap = overlap_manager.overlap(
                osc_bra,
                osc_ket,
                n_bra,
                n_plus,
                report_prefactor_as_exponent=False,
            )
            assert not isinstance(overlap, tuple)
            term_plus = np.sqrt(s_mu + 1) * overlap

        n_minus = shift_occupation(n_ket, mu, -1)
        term_minus: complex = 0.0 + 0.0j
        if n_minus is not None:
            overlap = overlap_manager.overlap(
                osc_bra,
                osc_ket,
                n_bra,
                n_minus,
                report_prefactor_as_exponent=False,
            )
            assert not isinstance(overlap, tuple)
            term_minus = np.sqrt(s_mu) * overlap

        val += coeff * prefactor * (term_plus - term_minus)

    return val


def q2_operator_element(
    osc_bra: LocalHarmonicOscillator,
    osc_ket: LocalHarmonicOscillator,
    n_bra: Sequence[int] | Occupation,
    n_ket: Sequence[int] | Occupation,
    mode_index: int,
    overlap_manager: OverlapManager,
) -> complex:
    n_bra_t = tuple(n_bra)
    n_ket_t = tuple(n_ket)

    mu = mode_index
    freq = osc_ket.frequencies[mu]
    s_mu = n_ket_t[mu]
    prefactor = -freq / 2.0

    base_overlap = overlap_manager.overlap(
        osc_bra,
        osc_ket,
        n_bra_t,
        n_ket_t,
        report_prefactor_as_exponent=False,
    )
    assert not isinstance(base_overlap, tuple)
    term_diag = (2 * s_mu + 1) * base_overlap

    term_minus: complex = 0.0 + 0.0j
    if s_mu >= 2:
        n_minus = shift_occupation(n_ket_t, mu, -2)
        assert n_minus is not None
        overlap_minus = overlap_manager.overlap(
            osc_bra,
            osc_ket,
            n_bra_t,
            n_minus,
            report_prefactor_as_exponent=False,
        )
        assert not isinstance(overlap_minus, tuple)
        term_minus = np.sqrt(s_mu * (s_mu - 1)) * overlap_minus

    n_plus = shift_occupation(n_ket_t, mu, +2)
    assert n_plus is not None
    overlap_plus = overlap_manager.overlap(
        osc_bra,
        osc_ket,
        n_bra_t,
        n_plus,
        report_prefactor_as_exponent=False,
    )
    assert not isinstance(overlap_plus, tuple)
    term_plus = np.sqrt((s_mu + 1) * (s_mu + 2)) * overlap_plus

    return prefactor * (term_diag - term_minus - term_plus)


class OperatorMatrixBuilder(OverlapManager):
    def overlap_states(
        self, state1: OscillatorFockState, state2: OscillatorFockState
    ) -> complex:
        val = self.overlap(
            state1.oscillator,
            state2.oscillator,
            state1.occupations,
            state2.occupations,
            report_prefactor_as_exponent=False,
        )
        assert not isinstance(val, tuple)
        return val

    def n_op(
        self,
        state1: OscillatorFockState,
        state2: OscillatorFockState,
        charge_index: int,
    ) -> complex:
        return n_operator_element(
            state1.oscillator,
            state2.oscillator,
            state1.occupations,
            state2.occupations,
            charge_index,
            overlap_manager=self,
        )

    def q2_op(
        self,
        state1: OscillatorFockState,
        state2: OscillatorFockState,
        mode_index: int,
    ) -> complex:
        return q2_operator_element(
            state1.oscillator,
            state2.oscillator,
            state1.occupations,
            state2.occupations,
            mode_index,
            overlap_manager=self,
        )

    def q2_total_op(
        self,
        state1: OscillatorFockState,
        state2: OscillatorFockState,
    ) -> complex:
        val: complex = 0.0 + 0.0j
        osc = state2.oscillator

        for mu in range(len(osc.frequencies)):
            val += self.q2_op(state1, state2, mu)

        return val

    def exp_operator(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


__all__ = ["shift_occupation", "n_operator_element", "OperatorMatrixBuilder"]
