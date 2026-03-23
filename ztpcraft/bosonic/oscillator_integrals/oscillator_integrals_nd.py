import math
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from .gaussian_hermite_nd import (
    G0_tensor,
    apply_L,
    integrate_gaussian_hermite,
)

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LocalHarmonicOscillator:
    """
    Represents a single N-dimensional harmonic oscillator in global coordinates.

    Parameters
    ----------
    transform : (K, N) ndarray
        Matrix S mapping global coordinates θ → local normal modes:
            ξ = S (θ - center)

    frequencies : (K,) ndarray
        Mode frequencies ω_k.

    center : (N,) ndarray
        Center position θ0.
    """

    transform: Array2D
    frequencies: Array1D
    center: Array1D

    def quadratic_form(self) -> Array2D:
        """Return K = S^T diag(ω) S"""
        return self.transform.T @ np.diag(self.frequencies) @ self.transform

    @property
    def num_modes(self) -> int:
        return self.transform.shape[0]

    @property
    def dim(self) -> int:
        return self.transform.shape[1]


@dataclass(frozen=True)
class OscillatorPairOverlapData:
    """
    Precomputed data needed to evaluate overlaps between two fixed local
    multidimensional harmonic oscillators while sweeping over different Fock
    occupations.

    This object stores the integral in the standard form

        ∫ d^Nθ exp[-1/2 (θ^T A θ + B^T θ + C)]
            × ∏_k H_{n_k}( D_k^T (θ - x0_k) )

    where the Hermite factors come from the two local oscillator wavefunctions.

    Attributes
    ----------
    gaussian_quadratic : (N, N) ndarray
        The matrix A in the Gaussian exponent.

    gaussian_linear : (N,) ndarray
        The vector B in the Gaussian exponent.

    gaussian_constant : float
        The scalar C in the Gaussian exponent.

    hermite_directions : list[(N,) ndarray]
        The list of linear directions D_k entering the Hermite arguments.
        By construction, the first block comes from oscillator 1 and the second
        block comes from oscillator 2.

    hermite_centers : list[(N,) ndarray]
        The list of displacement vectors x0_k used by the integration engine,
        so that each Hermite argument is represented as D_k^T (θ - x0_k).

        Note:
        The integration engine expects full displacement vectors x0_k, not just
        the scalar projection D_k^T θ_center. This is because it internally
        computes D_k^T (μ - x0_k).

    num_modes_osc1 : int
        Number of normal modes in oscillator 1.

    num_modes_osc2 : int
        Number of normal modes in oscillator 2.

    transform_osc1 : (N, N) ndarray
        The coordinate transform S1 used for oscillator 1. Included for
        reference/debugging.

    transform_osc2 : (N, N) ndarray
        The coordinate transform S2 used for oscillator 2. Included for
        reference/debugging.

    freq_osc1 : (N,) ndarray
        Mode frequencies ω^(1) for oscillator 1.

    freq_osc2 : (N,) ndarray
        Mode frequencies ω^(2) for oscillator 2.

    center_osc1 : (N,) ndarray
        Center location θ1 of oscillator 1 in the original coordinate system.

    center_osc2 : (N,) ndarray
        Center location θ2 of oscillator 2 in the original coordinate system.
    """

    gaussian_quadratic: Array2D
    gaussian_linear: Array1D
    gaussian_constant: float
    hermite_directions: list[Array1D]
    hermite_centers: list[Array1D]
    num_modes_osc1: int
    num_modes_osc2: int
    transform_osc1: Array2D
    transform_osc2: Array2D
    freq_osc1: Array1D
    freq_osc2: Array1D
    center_osc1: Array1D
    center_osc2: Array1D


def prepare_oscillator_pair_overlap(
    transform_osc1: npt.ArrayLike,
    mode_freqs_osc1: npt.ArrayLike,
    center_osc1: npt.ArrayLike,
    transform_osc2: npt.ArrayLike,
    mode_freqs_osc2: npt.ArrayLike,
    center_osc2: npt.ArrayLike,
) -> OscillatorPairOverlapData:
    """
    Precompute all geometry-dependent data needed for overlaps between two fixed
    local multidimensional harmonic oscillators.

    Parameters
    ----------
    transform_osc1 : (N, N) array_like
        Linear coordinate transform S1 for oscillator 1. The local normal-mode
        coordinates are assumed to be

            ξ^(1) = S1 @ (θ - θ1),

        so that row i of S1 defines the linear form entering the i-th Hermite
        polynomial of oscillator 1.

    mode_freqs_osc1 : (N,) array_like
        Frequencies ω^(1) of oscillator 1 in its local normal-mode coordinates.

    center_osc1 : (N,) array_like
        Center location θ1 of oscillator 1 in the original coordinate system.

    transform_osc2 : (N, N) array_like
        Linear coordinate transform S2 for oscillator 2, with

            ξ^(2) = S2 @ (θ - θ2).

    mode_freqs_osc2 : (N,) array_like
        Frequencies ω^(2) of oscillator 2.

    center_osc2 : (N,) array_like
        Center location θ2 of oscillator 2 in the original coordinate system.

    Returns
    -------
    OscillatorPairOverlapData
        A frozen dataclass containing the Gaussian exponent parameters and the
        Hermite data needed by `integrate_gaussian_hermite`.

    Notes
    -----
    For a normalized multidimensional harmonic oscillator state of the form

        ψ_n(θ) ∝ ∏_i H_{n_i}(ξ_i) exp[-1/2 ∑_i ω_i ξ_i^2],

    with ξ = S(θ - θ0), the Gaussian part can be written in the original
    coordinates as

        exp[-1/2 (θ - θ0)^T K (θ - θ0)],

    where

        K = S^T diag(ω) S.

    For the overlap of two such states, the combined Gaussian is

        exp[-1/2 (θ^T A θ + B^T θ + C)],

    with

        A = K1 + K2,
        B = -2 (K1 θ1 + K2 θ2),
        C = θ1^T K1 θ1 + θ2^T K2 θ2.

    This function only prepares the shared geometry data. The Fock occupations
    should be supplied later when evaluating a specific overlap.
    """
    transform_osc1 = np.asarray(transform_osc1, dtype=float)
    transform_osc2 = np.asarray(transform_osc2, dtype=float)
    mode_freqs_osc1 = np.asarray(mode_freqs_osc1, dtype=float)
    mode_freqs_osc2 = np.asarray(mode_freqs_osc2, dtype=float)
    center_osc1 = np.asarray(center_osc1, dtype=float)
    center_osc2 = np.asarray(center_osc2, dtype=float)

    num_modes_osc1 = transform_osc1.shape[0]
    num_modes_osc2 = transform_osc2.shape[0]
    num_coordinates = transform_osc1.shape[1]

    assert transform_osc1.shape == (num_modes_osc1, num_coordinates)
    assert transform_osc2.shape == (num_modes_osc2, num_coordinates)
    assert mode_freqs_osc1.shape == (num_modes_osc1,)
    assert mode_freqs_osc2.shape == (num_modes_osc2,)
    assert center_osc1.shape == (num_coordinates,)
    assert center_osc2.shape == (num_coordinates,)

    # Quadratic forms associated with the two Gaussian envelopes:
    #
    #   exp[-1/2 (θ - θ_a)^T K_a (θ - θ_a)]
    #
    # with
    #
    #   K_a = S_a^T diag(ω_a) S_a.
    quadratic_form_osc1: Array2D = np.asarray(
        transform_osc1.T @ np.diag(mode_freqs_osc1) @ transform_osc1,
        dtype=np.float64,
    )
    quadratic_form_osc2: Array2D = np.asarray(
        transform_osc2.T @ np.diag(mode_freqs_osc2) @ transform_osc2,
        dtype=np.float64,
    )

    # Combined Gaussian exponent:
    #
    #   -1/2 [ θ^T A θ + B^T θ + C ]
    #
    # after multiplying the two oscillator envelopes together.
    gaussian_quadratic: Array2D = np.asarray(
        quadratic_form_osc1 + quadratic_form_osc2, dtype=np.float64
    )
    gaussian_linear: Array1D = np.asarray(
        -2.0 * (quadratic_form_osc1 @ center_osc1 + quadratic_form_osc2 @ center_osc2),
        dtype=np.float64,
    )
    gaussian_constant = float(
        center_osc1 @ quadratic_form_osc1 @ center_osc1
        + center_osc2 @ quadratic_form_osc2 @ center_osc2
    )

    # Hermite arguments:
    #
    #   row_i(S1)^T (θ - θ1),   row_j(S2)^T (θ - θ2)
    #
    # The integration engine expects each argument in the form
    #
    #   D_k^T (θ - x0_k),
    #
    # so we store D_k as a row of the corresponding transform and x0_k as the
    # full center vector of that oscillator.
    hermite_directions: list[Array1D] = []
    hermite_centers: list[Array1D] = []

    for mode_index in range(num_modes_osc1):
        hermite_directions.append(np.asarray(transform_osc1[mode_index], dtype=float))
        hermite_centers.append(center_osc1.copy())

    for mode_index in range(num_modes_osc2):
        hermite_directions.append(np.asarray(transform_osc2[mode_index], dtype=float))
        hermite_centers.append(center_osc2.copy())

    return OscillatorPairOverlapData(
        gaussian_quadratic=gaussian_quadratic,
        gaussian_linear=gaussian_linear,
        gaussian_constant=gaussian_constant,
        hermite_directions=hermite_directions,
        hermite_centers=hermite_centers,
        num_modes_osc1=num_modes_osc1,
        num_modes_osc2=num_modes_osc2,
        transform_osc1=transform_osc1,
        transform_osc2=transform_osc2,
        freq_osc1=mode_freqs_osc1,
        freq_osc2=mode_freqs_osc2,
        center_osc1=center_osc1,
        center_osc2=center_osc2,
    )


## Cache
@dataclass
class CachedOverlapInternals:
    """
    Cached Gaussian + Hermite internal quantities for repeated evaluations.
    """

    mu: Array1D
    Gamma: Array2D
    M: Array2D
    Z0_exponent: float


def build_overlap_cache(
    pair_data: OscillatorPairOverlapData,
) -> CachedOverlapInternals:
    """
    Precompute expensive quantities reused across many Fock evaluations.
    """
    A: Array2D = pair_data.gaussian_quadratic
    B: Array1D = pair_data.gaussian_linear
    D_list: list[Array1D] = pair_data.hermite_directions

    # Cholesky
    L = np.linalg.cholesky(A)

    def solve_A(b: Array1D) -> Array1D:
        return np.asarray(np.linalg.solve(L.T, np.linalg.solve(L, b)), dtype=np.float64)

    mu: Array1D = np.asarray(-0.5 * solve_A(B), dtype=np.float64)

    # prefactor
    logdetA = 2.0 * np.sum(np.log(np.diag(L)))
    BtAiB: float = float(B @ solve_A(B))

    Z0_exponent: float = float(
        A.shape[0] / 2 * np.log(2 * np.pi)
        + 0.125 * BtAiB
        - 0.5 * pair_data.gaussian_constant
        - 0.5 * logdetA
    )

    # build Gamma
    aux: Array2D = np.asarray(
        np.column_stack([solve_A(d) for d in D_list]), dtype=float
    )
    Gamma: Array2D = np.array(
        [
            [D_list[i] @ aux[:, j] for j in range(len(D_list))]
            for i in range(len(D_list))
        ],
        dtype=np.float64,
    )

    M: Array2D = np.asarray(2.0 * Gamma - np.eye(len(D_list)), dtype=np.float64)

    return CachedOverlapInternals(mu, Gamma, M, Z0_exponent)


def overlap_with_cache(
    pair_data: OscillatorPairOverlapData,
    cache: CachedOverlapInternals,
    occupations_osc1: Sequence[int],
    occupations_osc2: Sequence[int],
    *,
    use_logs: bool = False,
    report_prefactor_as_exponent: bool = False,
) -> complex | tuple[complex, float]:
    """
    Fast overlap evaluation using precomputed cache.
    """

    n1: npt.NDArray[np.int_] = np.asarray(occupations_osc1, dtype=int)
    n2: npt.NDArray[np.int_] = np.asarray(occupations_osc2, dtype=int)
    n_total: npt.NDArray[np.int_] = np.concatenate([n1, n2])

    D_list: list[Array1D] = pair_data.hermite_directions
    x0_list: list[Array1D] = pair_data.hermite_centers

    # m_k = D_k^T (mu - x0_k)
    m = np.array(
        [D @ (cache.mu - x0) for D, x0 in zip(D_list, x0_list)],
        dtype=complex,
    )

    G0: npt.NDArray[np.complex128] = G0_tensor(m, n_total, use_logs)
    p_max = int(n_total.sum()) // 2

    g_tensor = G0
    accum = G0.copy()

    for p in range(1, p_max + 1):
        g_tensor = np.asarray(apply_L(g_tensor, cache.M), dtype=np.complex128) / p
        accum += g_tensor

    coeff = accum[tuple([-1] * len(n_total))]

    prefactor = sum(math.lgamma(int(n) + 1) for n in n_total)

    if np.isclose(coeff, 0.0, atol=1e-14, rtol=0.0):
        if report_prefactor_as_exponent:
            return complex(0.0, 0.0), -np.inf
        return 0.0 + 0.0j

    total_exponent = prefactor + cache.Z0_exponent

    if report_prefactor_as_exponent:
        return coeff, total_exponent

    return np.exp(np.log(coeff) + total_exponent)


def overlap_between_oscillator_fock_states(
    pair_data: OscillatorPairOverlapData,
    occupations_osc1: Sequence[int],
    occupations_osc2: Sequence[int],
    *,
    use_logs: bool = False,
    include_normalization: bool = True,
    report_prefactor_as_exponent: bool = False,
) -> complex | tuple[complex, complex]:
    """
    Evaluate overlap between two local multidimensional oscillator Fock states.
    """
    occupations_osc1_arr: npt.NDArray[np.int_] = np.asarray(occupations_osc1, dtype=int)
    occupations_osc2_arr: npt.NDArray[np.int_] = np.asarray(occupations_osc2, dtype=int)

    assert occupations_osc1_arr.shape == (pair_data.num_modes_osc1,)
    assert occupations_osc2_arr.shape == (pair_data.num_modes_osc2,)
    assert np.all(occupations_osc1_arr >= 0)
    assert np.all(occupations_osc2_arr >= 0)

    combined_occupations = np.concatenate([occupations_osc1_arr, occupations_osc2_arr])
    integral_result = integrate_gaussian_hermite(
        covariance_matrix=pair_data.gaussian_quadratic,
        linear_term=pair_data.gaussian_linear,
        constant_term=pair_data.gaussian_constant,
        hermite_directions=pair_data.hermite_directions,
        displacement_vectors=pair_data.hermite_centers,
        hermite_orders=combined_occupations,
        use_logs=use_logs,
        report_prefactor_as_exponent=report_prefactor_as_exponent,
    )

    if not include_normalization:
        return integral_result

    normalization_log = 0.0
    for omega, n in zip(pair_data.freq_osc1, occupations_osc1_arr):
        normalization_log += 0.25 * np.log(omega / np.pi)
        normalization_log -= 0.5 * (int(n) * np.log(2.0) + math.lgamma(int(n) + 1))

    for omega, n in zip(pair_data.freq_osc2, occupations_osc2_arr):
        normalization_log += 0.25 * np.log(omega / np.pi)
        normalization_log -= 0.5 * (int(n) * np.log(2.0) + math.lgamma(int(n) + 1))

    if report_prefactor_as_exponent:
        assert isinstance(integral_result, tuple)
        coeff, exponent = integral_result
        return coeff, exponent + normalization_log

    return integral_result * np.exp(normalization_log)


class OscillatorOverlapEngine:
    """
    High-level interface for computing overlaps between two oscillators.

    Combines:
    - geometry (pair_data)
    - cache (optional)
    - evaluation methods
    """

    def __init__(
        self,
        osc1: LocalHarmonicOscillator,
        osc2: LocalHarmonicOscillator,
        *,
        use_cache: bool = True,
    ) -> None:
        self.osc1 = osc1
        self.osc2 = osc2

        # build geometry
        self.pair_data = prepare_oscillator_pair_overlap(
            osc1.transform,
            osc1.frequencies,
            osc1.center,
            osc2.transform,
            osc2.frequencies,
            osc2.center,
        )

        # optionally build cache
        self.cache: CachedOverlapInternals | None = (
            build_overlap_cache(self.pair_data) if use_cache else None
        )

    def overlap(
        self,
        n1: Sequence[int],
        n2: Sequence[int],
        *,
        use_logs: bool = False,
        include_normalization: bool = True,
        report_prefactor_as_exponent: bool = False,
    ) -> complex | tuple[complex, complex | float]:
        """
        Compute overlap ⟨n1 | n2⟩
        """

        if self.cache is not None:
            val = overlap_with_cache(
                self.pair_data,
                self.cache,
                n1,
                n2,
                use_logs=use_logs,
                report_prefactor_as_exponent=report_prefactor_as_exponent,
            )
        else:
            val = overlap_between_oscillator_fock_states(
                self.pair_data,
                n1,
                n2,
                use_logs=use_logs,
                include_normalization=False,
                report_prefactor_as_exponent=report_prefactor_as_exponent,
            )

        if not include_normalization:
            return val

        if report_prefactor_as_exponent:
            assert isinstance(val, tuple)
            coeff, exponent = val
            return coeff, exponent + self._normalization(n1, n2)

        assert not isinstance(val, tuple)
        return val * self._normalization(n1, n2)

    def _normalization(self, n1: Sequence[int], n2: Sequence[int]) -> float:
        """HO normalization factors"""
        log_norm: float = 0.0

        for w, n in zip(self.osc1.frequencies, n1):
            log_norm += 0.25 * np.log(w / np.pi)
            log_norm -= 0.5 * (int(n) * np.log(2.0) + math.lgamma(int(n) + 1))

        for w, n in zip(self.osc2.frequencies, n2):
            log_norm += 0.25 * np.log(w / np.pi)
            log_norm -= 0.5 * (int(n) * np.log(2.0) + math.lgamma(int(n) + 1))

        return np.exp(log_norm)

    def batch(
        self,
        pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[complex | tuple[complex, complex | float]]:
        """Batch evaluation"""
        return [self.overlap(n1, n2) for n1, n2 in pairs]


def batched_overlaps(
    pair_data: OscillatorPairOverlapData,
    occupation_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    use_logs: bool = False,
    include_normalization: bool = True,
) -> list[complex | tuple[complex, complex]]:
    """
    Evaluate overlaps for many occupation pairs using the same fixed geometry.
    """
    return [
        overlap_between_oscillator_fock_states(
            pair_data,
            occ1,
            occ2,
            use_logs=use_logs,
            include_normalization=include_normalization,
        )
        for occ1, occ2 in occupation_pairs
    ]


__all__ = [
    "LocalHarmonicOscillator",
    "OscillatorPairOverlapData",
    "CachedOverlapInternals",
    "prepare_oscillator_pair_overlap",
    "build_overlap_cache",
    "overlap_with_cache",
    "overlap_between_oscillator_fock_states",
    "batched_overlaps",
    "OscillatorOverlapEngine",
]
