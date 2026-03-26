import numpy as np

from ztpcraft.bosonic.oscillator_integrals.oscillator_operators import (
    q2_operator_element,
)
from ztpcraft.bosonic.oscillator_integrals.oscillator_overlap_cache import OverlapManager
from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator


def make_1d_oscillator(freq: float) -> LocalHarmonicOscillator:
    a = 3.4
    return LocalHarmonicOscillator(
        transform_xi_theta=np.array([[a]]),
        transform_qn=np.array([[1 / a]]),
        frequencies=np.array([freq]),
        center=np.array([0.32]),
    )


def test_overlap_identity_and_orthogonality() -> None:
    osc = make_1d_oscillator(3.24)
    manager = OverlapManager()

    for n in range(5):
        for m in range(5):
            val = manager.overlap(osc, osc, (n,), (m,))

            if n == m:
                assert np.allclose(val, 1.0, atol=1e-10)
            else:
                assert np.allclose(val, 0.0, atol=1e-10)


def test_overlap_ground_state_normalization() -> None:
    osc = make_1d_oscillator(3.24)
    manager = OverlapManager()

    val = manager.overlap(osc, osc, (0,), (0,))
    assert np.allclose(val, 1.0, atol=1e-10)


def q2_analytic(n: int, m: int, omega: float) -> complex:
    if n == m:
        return -omega / 2 * (2 * n + 1)
    if n == m + 2:
        return -omega / 2 * (-np.sqrt(n * (n - 1)))
    if n == m - 2:
        return -omega / 2 * (-np.sqrt((n + 1) * (n + 2)))
    return 0.0


def test_q2_operator() -> None:
    omega = 1.7
    osc = make_1d_oscillator(omega)
    manager = OverlapManager()

    for n in range(5):
        for m in range(5):
            val = q2_operator_element(
                osc,
                osc,
                (n,),
                (m,),
                mode_index=0,
                overlap_manager=manager,
            )
            expected = q2_analytic(n, m, omega)
            assert np.allclose(val, expected, atol=1e-10)


def test_q2_hermiticity() -> None:
    osc = make_1d_oscillator(1.3)
    manager = OverlapManager()

    for n in range(4):
        for m in range(4):
            v1 = q2_operator_element(osc, osc, (n,), (m,), 0, manager)
            v2 = q2_operator_element(osc, osc, (m,), (n,), 0, manager)
            assert np.allclose(v1, np.conjugate(v2))


def q_analytic(n: int, m: int, omega: float) -> complex:
    if n == m + 1:
        return 1j * np.sqrt(omega / 2) * np.sqrt(n)
    if n == m - 1:
        return -1j * np.sqrt(omega / 2) * np.sqrt(n + 1)
    return 0.0
