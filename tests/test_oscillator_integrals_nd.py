import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
import scipy as sp
from scipy import integrate

from ztpcraft.bosonic.oscillator_integrals.gaussian_hermite_nd import (
    integrate_gaussian_hermite,
)
from ztpcraft.bosonic.oscillator_integrals.oscillator_integrals_nd import (
    CachedOverlapInternals,
    LocalHarmonicOscillator,
    OscillatorOverlapEngine,
    OscillatorPairOverlapData,
    batched_overlaps,
    build_overlap_cache,
    overlap_between_oscillator_fock_states,
    overlap_with_cache,
    prepare_oscillator_pair_overlap,
)

Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def integrate_nd_nquad(
    f: Callable[..., float],
    n: int,
    epsabs: float = 1e-8,
    epsrel: float = 1e-6,
) -> tuple[float, float]:
    """Integrate f(x1,...,xn) over R^n using scipy.integrate.nquad."""
    bounds = [(-np.inf, np.inf)] * n

    # nquad expects f(xn, ..., x1)
    def f_rev(*xs: np.float64) -> float:
        return float(f(*xs[::-1]))

    val, err = integrate.nquad(f_rev, bounds, opts={"epsabs": epsabs, "epsrel": epsrel})
    return float(val), float(err)


def gaussian_hermite_nd(
    *x: float,
    A: Array2D,
    B: Array1D,
    C: float,
    D_list: Sequence[Array1D],
    x0_list: Sequence[Array1D],
    s_list: Sequence[int],
) -> float:
    x_vec: Array1D = np.asarray(x, dtype=np.float64)
    quad_form = float(x_vec @ A @ x_vec + B @ x_vec + C)
    exp_factor = math.exp(-0.5 * quad_form)
    prod_factor = 1.0
    for D_k, x0_k, n_k in zip(D_list, x0_list, s_list):
        argument = float(D_k @ (x_vec - x0_k))
        prod_factor *= float(sp.special.eval_hermite(int(n_k), argument))
    return exp_factor * prod_factor


def gaussian_hermite_cos_nd(
    *x: float,
    A: Array2D,
    B: Array1D,
    C: float,
    D_list: Sequence[Array1D],
    x0_list: Sequence[Array1D],
    cos_arg_list: Array1D,
    s_list: Sequence[int],
) -> float:
    x_vec: Array1D = np.asarray(x, dtype=np.float64)
    cos_vec: Array1D = np.asarray(cos_arg_list, dtype=np.float64)
    quad_form = float(x_vec @ A @ x_vec + B @ x_vec + C)
    exp_factor = math.exp(-0.5 * quad_form)
    prod_factor = 1.0
    for D_k, x0_k, n_k in zip(D_list, x0_list, s_list):
        argument = float(D_k @ (x_vec - x0_k))
        prod_factor *= float(sp.special.eval_hermite(int(n_k), argument))
    cos_term = math.cos(float(np.sum(cos_vec * x_vec)))
    return exp_factor * prod_factor * cos_term


def test_1d_ground_state_integral_matches_closed_form():
    A = np.array([[2.0]])
    B = np.array([0.0])
    C = 0.0
    I: complex = integrate_gaussian_hermite(A, B, C, [], [], [])
    expected = math.sqrt(math.pi)
    assert np.isclose(I, expected, atol=1e-10)


def test_1d_single_hermite_n2_is_zero():
    A = np.array([[2.0]])
    B = np.array([0.0])
    C = 0.0
    D_list = [np.array([1.0])]
    x0_list = [np.array([0.0])]
    s_list = [2]
    I: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)
    assert np.isclose(I, 0.0, atol=1e-10)


def test_1d_two_n1_hermites_matches_closed_form():
    A = np.array([[2.0]])
    B = np.array([0.0])
    C = 0.0
    D_list = [np.array([1.0]), np.array([1.0])]
    x0_list = [np.array([0.0]), np.array([0.0])]
    s_list = [1, 1]
    I: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)
    expected = 2.0 * math.sqrt(math.pi)
    assert np.isclose(I, expected, atol=1e-10)


def test_1d_high_odd_total_degree_is_zero():
    A = np.array([[2.0]])
    B = np.array([0.0])
    C = 0.0
    D_list = [np.array([1.0]), np.array([1.0])]
    x0_list = [np.array([0.0]), np.array([0.0])]
    s_list = [20, 19]  # odd total degree -> odd integrand
    I: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)
    assert np.isclose(I, 0.0, atol=1e-10)


def test_1d_two_n3_hermites_matches_closed_form():
    A = np.array([[2.0]])
    B = np.array([0.0])
    C = 0.0
    D_list = [np.array([1.0]), np.array([1.0])]
    x0_list = [np.array([0.0]), np.array([0.0])]
    s_list = [3, 3]
    I: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)
    expected = math.sqrt(math.pi) * (2**3) * math.factorial(3)
    assert np.isclose(I, expected, atol=1e-10)


def test_2d_axis_separated_odd_orders_is_zero():
    A = np.array([[2.0, 0.0], [0.0, 2.0]])
    B = np.array([0.0, 0.0])
    C = 0.0
    D_list = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    x0_list = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    s_list = [3, 3]
    I: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)
    assert np.isclose(I, 0.0, atol=1e-10)


@pytest.mark.slow
def test_2d_general_case_matches_nquad():
    A = np.array([[2.0, 0.0], [0.0, 2.0]])
    B = np.array([0.0, 0.0])
    C = 0.0
    D_list = [np.array([1.0, 2.0]), np.array([3.0, 1.0])]
    x0_list = [np.array([1.0, 1.5]), np.array([0.0, 0.2])]
    s_list = [3, 2]

    def f(x1: float, x2: float) -> float:
        return gaussian_hermite_nd(
            x1, x2, A=A, B=B, C=C, D_list=D_list, x0_list=x0_list, s_list=s_list
        )

    I_num, _ = integrate_nd_nquad(f, 2)
    I_mgf: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)

    assert np.isclose(I_num, np.real(I_mgf), atol=1e-10)
    assert np.isclose(np.imag(I_mgf), 0.0, atol=1e-10)


@pytest.mark.slow
def test_2d_general_case_with_linear_term_matches_nquad():
    A = np.array([[1.1, 0.1], [0.1, 1.1]])
    B = np.array([0.4, -0.1])
    C = 0.5
    D_list = [np.array([1.0, 2.0]), np.array([3.0, 1.0])]
    x0_list = [np.array([1.0, 1.5]), np.array([0.0, 0.2])]
    s_list = [3, 2]

    def f(x1: float, x2: float) -> float:
        return gaussian_hermite_nd(
            x1, x2, A=A, B=B, C=C, D_list=D_list, x0_list=x0_list, s_list=s_list
        )

    I_num, _ = integrate_nd_nquad(f, 2)
    I_mgf: complex = integrate_gaussian_hermite(A, B, C, D_list, x0_list, s_list)

    assert np.isclose(I_num, np.real(I_mgf), atol=1e-10)
    assert np.isclose(np.imag(I_mgf), 0.0, atol=1e-10)


@pytest.mark.slow
def test_2d_general_case_with_cosine_matches_complex_shift_trick():
    A = np.array([[1.1, 0.1], [0.1, 1.1]])
    B = np.array([0.4, -0.1])
    C = 0.5
    D_list = [np.array([1.0, 2.0]), np.array([3.0, 1.0])]
    x0_list = [np.array([1.0, 1.5]), np.array([0.0, 0.2])]
    s_list = [3, 2]
    cos_arg_list = np.array([1.0, 1.0])
    B_complex = B - 2j * cos_arg_list

    def f(x1: float, x2: float) -> float:
        return gaussian_hermite_cos_nd(
            x1,
            x2,
            A=A,
            B=B,
            C=C,
            D_list=D_list,
            x0_list=x0_list,
            cos_arg_list=cos_arg_list,
            s_list=s_list,
        )

    I_num, _ = integrate_nd_nquad(f, 2)
    I_mgf: float = float(
        np.real(integrate_gaussian_hermite(A, B_complex, C, D_list, x0_list, s_list))
    )

    assert np.isclose(I_num, I_mgf, atol=1e-10)


def _build_sample_oscillators() -> (
    tuple[LocalHarmonicOscillator, LocalHarmonicOscillator]
):
    transform_osc1: Array2D = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    frequencies_osc1: Array1D = np.array([1.3, 0.9], dtype=np.float64)
    center_osc1: Array1D = np.array([0.2, -0.3], dtype=np.float64)

    transform_osc2: Array2D = np.array([[1.1, 0.2], [-0.15, 0.95]], dtype=np.float64)
    frequencies_osc2: Array1D = np.array([1.1, 1.7], dtype=np.float64)
    center_osc2: Array1D = np.array([-0.1, 0.4], dtype=np.float64)

    osc1 = LocalHarmonicOscillator(
        transform=transform_osc1,
        frequencies=frequencies_osc1,
        center=center_osc1,
    )
    osc2 = LocalHarmonicOscillator(
        transform=transform_osc2,
        frequencies=frequencies_osc2,
        center=center_osc2,
    )
    return osc1, osc2


def _build_pair_and_cache() -> tuple[OscillatorPairOverlapData, CachedOverlapInternals]:
    osc1, osc2 = _build_sample_oscillators()
    pair_data = prepare_oscillator_pair_overlap(
        osc1.transform,
        osc1.frequencies,
        osc1.center,
        osc2.transform,
        osc2.frequencies,
        osc2.center,
    )
    cache = build_overlap_cache(pair_data)
    return pair_data, cache


def test_overlap_cache_matches_direct_without_normalization():
    pair_data, cache = _build_pair_and_cache()
    occupations = [
        ((0, 0), (0, 0)),
        ((1, 0), (0, 1)),
        ((2, 1), (1, 0)),
        ((3, 0), (2, 1)),
    ]

    for occ1, occ2 in occupations:
        val_direct = overlap_between_oscillator_fock_states(
            pair_data,
            occ1,
            occ2,
            include_normalization=False,
        )
        val_cache = overlap_with_cache(pair_data, cache, occ1, occ2)
        assert np.isclose(val_direct, val_cache, atol=1e-11)


def test_overlap_with_cache_prefactor_roundtrip():
    pair_data, cache = _build_pair_and_cache()
    occ1 = (2, 1)
    occ2 = (1, 2)

    val = overlap_with_cache(pair_data, cache, occ1, occ2)
    coeff_exp = overlap_with_cache(
        pair_data,
        cache,
        occ1,
        occ2,
        report_prefactor_as_exponent=True,
    )
    assert isinstance(coeff_exp, tuple)
    coeff, exponent = coeff_exp
    reconstructed = coeff * np.exp(exponent)

    assert np.isclose(val, reconstructed, atol=1e-11)


def test_engine_cache_toggle_matches_for_overlaps():
    osc1, osc2 = _build_sample_oscillators()
    engine_cache = OscillatorOverlapEngine(osc1, osc2, use_cache=True)
    engine_no_cache = OscillatorOverlapEngine(osc1, osc2, use_cache=False)

    occupations = [
        ((0, 0), (0, 0)),
        ((1, 1), (1, 0)),
        ((2, 0), (0, 2)),
        ((3, 1), (2, 1)),
    ]

    for occ1, occ2 in occupations:
        val_cache = engine_cache.overlap(occ1, occ2, include_normalization=False)
        val_no_cache = engine_no_cache.overlap(occ1, occ2, include_normalization=False)
        assert np.isclose(val_cache, val_no_cache, atol=1e-11)

        val_cache_norm = engine_cache.overlap(occ1, occ2, include_normalization=True)
        val_no_cache_norm = engine_no_cache.overlap(
            occ1, occ2, include_normalization=True
        )
        assert np.isclose(val_cache_norm, val_no_cache_norm, atol=1e-11)


def test_batched_overlaps_matches_manual_loop():
    pair_data, _ = _build_pair_and_cache()
    occupation_pairs = [
        ((0, 0), (0, 0)),
        ((1, 0), (0, 1)),
        ((1, 1), (1, 1)),
        ((2, 1), (1, 0)),
    ]

    batch_vals = batched_overlaps(
        pair_data,
        occupation_pairs,
        include_normalization=True,
    )
    loop_vals = [
        overlap_between_oscillator_fock_states(
            pair_data,
            occ1,
            occ2,
            include_normalization=True,
        )
        for occ1, occ2 in occupation_pairs
    ]

    for val_batch, val_loop in zip(batch_vals, loop_vals):
        assert np.isclose(val_batch, val_loop, atol=1e-11)
