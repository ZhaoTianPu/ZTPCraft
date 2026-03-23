import math
from typing import Any, Literal, overload
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


@overload
def G0_tensor(
    m: npt.ArrayLike, n_list: npt.ArrayLike, use_logs: Literal[False] = False
) -> npt.NDArray[np.complex128]: ...


@overload
def G0_tensor(
    m: npt.ArrayLike, n_list: npt.ArrayLike, use_logs: Literal[True]
) -> npt.NDArray[np.complex128]: ...


def G0_tensor(
    m: npt.ArrayLike, n_list: npt.ArrayLike, use_logs: bool = False
) -> npt.NDArray[np.complex128]:
    """
    Return tensor G0[alpha_1,...,alpha_K] = Π_k (2 m_k)^{alpha_k} / alpha_k!
    shape = (n_1+1)*(n_2+1)*...*(n_K+1)
    """
    m = np.asarray(m, complex)
    n_list = np.asarray(n_list, int)
    vecs: list[npt.NDArray[np.complex128]] = []
    for k, nk in enumerate(n_list):
        r = np.arange(int(nk) + 1, dtype=int)

        if use_logs:
            # numerically stable: exp(r*log|2 m_k| - log Γ(r+1)) with sign
            if m[k] == 0.0:
                v = np.zeros_like(r, dtype=complex)
                v[0] = 1.0
            else:
                log_fact = np.array([math.lgamma(int(i) + 1) for i in r], dtype=float)
                v = np.exp(r * np.log(abs(2.0 * m[k])) - log_fact)
                # apply sign^r without pow for speed
                v *= (1 - 2 * (m[k] < 0)) ** r
        else:
            # simple and fast (fine for small degrees ≤ ~20)
            fact = np.array([math.factorial(int(i)) for i in r], float)
            v = (2.0 * m[k]) ** r / fact

        vecs.append(np.asarray(v, dtype=np.complex128))

    # K-way outer product
    if not vecs:
        return np.asarray(1.0 + 0.0j, dtype=np.complex128)
    g0_tensor = vecs[0]
    for vec in vecs[1:]:
        g0_tensor = np.multiply.outer(g0_tensor, vec)
    return g0_tensor


def apply_L(
    tensor: npt.ArrayLike, matrix: npt.ArrayLike
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """
    Apply the degree-raising operator L defined by:
      L[T] = sum_i M_ii * S_ii[T] + sum_{i<j} 2*M_ij * S_ij[T]
    where S_ii shifts +2 along axis i, and S_ij shifts +1 along axes i and j.

    Parameters
    ----------
    tensor : ndarray
        K-way coefficient tensor with shape (n1+1, n2+1, ..., nK+1).
    M : (K,K) array_like
        Symmetric matrix (M = 2*Gamma - I). K must equal tensor.ndim.

    Returns
    -------
    mapped_tensor : ndarray
        Tensor with the same shape as `tensor`, containing L[tensor].
    """
    tensor = np.asarray(tensor)
    matrix = np.asarray(matrix)
    K = tensor.ndim
    assert matrix.shape == (K, K), "M must be KxK where K = tensor.ndim"

    out = np.zeros_like(tensor)

    # Diagonal shifts: +2 on axis i
    for i in range(K):
        mii = matrix[i, i]
        if mii == 0:
            continue
        n_i = tensor.shape[i]
        if n_i <= 2:
            continue  # nothing to shift by +2
        src = [slice(None)] * K
        dst = [slice(None)] * K
        src[i] = slice(0, n_i - 2)
        dst[i] = slice(2, n_i)
        out[tuple(dst)] += mii * tensor[tuple(src)]

    # Off-diagonal shifts: +1 on axes i and j (i < j)
    for i in range(K):
        n_i = tensor.shape[i]
        if n_i <= 1:
            continue
        for j in range(i + 1, K):
            mij = matrix[i, j]
            if mij == 0:
                continue
            n_j = tensor.shape[j]
            if n_j <= 1:
                continue
            src = [slice(None)] * K
            dst = [slice(None)] * K
            src[i] = slice(0, n_i - 1)
            src[j] = slice(0, n_j - 1)
            dst[i] = slice(1, n_i)
            dst[j] = slice(1, n_j)
            out[tuple(dst)] += 2.0 * mij * tensor[tuple(src)]

    return out


@overload
def integrate_gaussian_hermite(
    covariance_matrix: npt.ArrayLike,
    linear_term: npt.ArrayLike,
    constant_term: complex | float,
    hermite_directions: Sequence[npt.ArrayLike],
    displacement_vectors: Sequence[npt.ArrayLike],
    hermite_orders: Sequence[int] | npt.NDArray[np.integer[Any]],
    use_logs: bool = False,
    report_prefactor_as_exponent: Literal[False] = False,
) -> complex: ...


@overload
def integrate_gaussian_hermite(
    covariance_matrix: npt.ArrayLike,
    linear_term: npt.ArrayLike,
    constant_term: complex | float,
    hermite_directions: Sequence[npt.ArrayLike],
    displacement_vectors: Sequence[npt.ArrayLike],
    hermite_orders: Sequence[int] | npt.NDArray[np.integer[Any]],
    use_logs: bool,
    report_prefactor_as_exponent: Literal[True],
) -> tuple[complex, complex]: ...


def integrate_gaussian_hermite(
    covariance_matrix: npt.ArrayLike,
    linear_term: npt.ArrayLike,
    constant_term: complex | float,
    hermite_directions: Sequence[npt.ArrayLike],
    displacement_vectors: Sequence[npt.ArrayLike],
    hermite_orders: Sequence[int] | npt.NDArray[np.integer[Any]],
    use_logs: bool = False,
    report_prefactor_as_exponent: bool = False,
) -> complex | tuple[complex, complex]:
    """
    Compute the integral:
    I = int exp[-1/2 (x^T A x + B^T x + C)] * prod_k H_{s_k}( D_k^T (x - x0_k) ) dx
    for physicists' Hermite H_n, with distinct linear forms D_k.
    Here,
    A = covariance_matrix
    B = linear_term
    C = constant_term
    D_k = hermite_directions[k]
    x0_k = displacement_vectors[k]
    s_k = hermite_orders[k]
    Let N be the number of dimensions of the system, K be the total number of Hermite polynomials of nonzero order in the product, then the size of each input array is:
    - covariance_matrix: (N,N)
    - linear_term: (N,)
    - constant_term: float
    - hermite_directions: list of (N,) ndarrays, length K
    - displacement_vectors: list of (N,) ndarrays, length K
    - hermite_orders: list/array of nonnegative ints, length K

    Parameters
    ----------
    covariance_matrix : (N,N) ndarray, positive-definite, real
    linear_term : (N,) ndarray, complex
    constant_term : float, complex
    hermite_directions : list of (N,) ndarrays, length K
    displacement_vectors : list of (N,) ndarrays, length K
    hermite_orders : list/array of nonnegative ints, length K

    Returns
    -------
    I : float
    """

    # ---- dimensions / checks
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)
    linear_term = np.asarray(linear_term, dtype=complex)
    N = covariance_matrix.shape[0]
    assert covariance_matrix.shape == (N, N)
    assert linear_term.shape == (N,)
    K = len(hermite_directions)  # total number of Hermite polynomials
    assert len(displacement_vectors) == K
    assert len(hermite_orders) == K
    hermite_orders = np.asarray(hermite_orders, dtype=int)
    assert np.all(hermite_orders >= 0)
    n_tot = int(hermite_orders.sum())

    # Cholesky for stability
    L = np.linalg.cholesky(covariance_matrix)

    # Solve A x = b via Cholesky
    def solve_A(
        b: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        return np.linalg.solve(L.T, np.linalg.solve(L, b))

    # μ = -1/2 A^{-1} B
    mu = -0.5 * solve_A(linear_term)

    # log(det A) from Cholesky
    logdetA = 2.0 * np.sum(np.log(np.diag(L)))

    # Prefactor Z0
    BtAiB = linear_term @ solve_A(linear_term)
    Z0_exponent = (
        N / 2 * np.log(2 * np.pi) + 0.125 * BtAiB - 0.5 * constant_term - 0.5 * logdetA
    )
    # Z0 = np.exp(Z0_exponent)

    # if there is no excitation at all, return Z0
    if K == 0 or np.all(np.asarray(hermite_orders, int) == 0):
        if report_prefactor_as_exponent:
            return 1, Z0_exponent
        else:
            Z0 = np.exp(Z0_exponent)
            return Z0

    # Build m_k and Γ_{ij} without forming A^{-1}
    # m_k = D_k^T (μ - x0_k)
    hermite_directions = [np.asarray(d, dtype=float) for d in hermite_directions]
    displacement_vectors = [
        np.asarray(x_0, dtype=float) for x_0 in displacement_vectors
    ]
    m = np.array(
        [D @ (mu - x_0) for D, x_0 in zip(hermite_directions, displacement_vectors)],
        dtype=complex,
    )

    # Γ_{ij} = D_i^T A^{-1} D_j = D_i^T y_j,  where A y_j = D_j
    # Solve once per j
    aux_vectors = np.column_stack([solve_A(d) for d in hermite_directions])  # (N,K)
    Gamma = np.array(
        [
            [hermite_directions[i] @ aux_vectors[:, j] for j in range(K)]
            for i in range(K)
        ],
        dtype=float,
    )

    # M = 2 Γ - I
    M = 2.0 * Gamma - np.eye(K)

    # Multivariate series coefficient of exp(2 t^T m) * exp(t^T M t)
    # For a given multi-index n_list, want to find the coefficient of
    # t_1^n_1 * t_2^n_2 * ... * t_K^n_K
    # Represent a K-variate polynomial as a dict { multi_index_tuple : coeff }
    # with per-variable degree cap n_list and total degree cap n_tot.

    # max_deg = tuple(int(n) for n in n_list)

    G0 = G0_tensor(m, hermite_orders, use_logs)
    n_tot = int(hermite_orders.sum())
    p_max = n_tot // 2

    G_tensor_list = [G0]

    def L_p_operator(
        G_p_minus_1_tensor: npt.NDArray[np.complex128],
        p: int,
        matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        if p == 0:
            return G_p_minus_1_tensor
        else:
            return (
                np.asarray(apply_L(G_p_minus_1_tensor, matrix), dtype=np.complex128) / p
            )

    for p in range(1, p_max + 1):
        G_tensor_list.append(L_p_operator(G_tensor_list[-1], p, M))

    # F_tensor is the sum of all G_tensors
    F_tensor = np.sum(G_tensor_list, axis=0)

    # read off the coefficient
    coeff = F_tensor[tuple([-1] * len(hermite_orders))]

    # multiply by a prefactor prod_k n_k!
    prefactor_exponent = sum(math.lgamma(int(n) + 1) for n in hermite_orders)
    # prefactor = np.exp(prefactor_exponent)

    # Final value
    if report_prefactor_as_exponent:
        return coeff, Z0_exponent + prefactor_exponent
    else:
        log_value: complex = np.log(coeff) + prefactor_exponent + Z0_exponent
        return np.exp(log_value)


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


@overload
def overlap_between_oscillator_fock_states(
    pair_data: OscillatorPairOverlapData,
    occupations_osc1: Sequence[int],
    occupations_osc2: Sequence[int],
    *,
    use_logs: bool = False,
    include_normalization: bool = True,
    report_prefactor_as_exponent: Literal[False] = False,
) -> complex: ...


@overload
def overlap_between_oscillator_fock_states(
    pair_data: OscillatorPairOverlapData,
    occupations_osc1: Sequence[int],
    occupations_osc2: Sequence[int],
    *,
    use_logs: bool,
    include_normalization: bool = True,
    report_prefactor_as_exponent: Literal[True],
) -> tuple[complex, complex]: ...


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
    Evaluate the overlap between two local multidimensional oscillator Fock states
    for a fixed precomputed oscillator pair.

    Parameters
    ----------
    pair_data : OscillatorPairOverlapData
        Precomputed geometry-dependent overlap data returned by
        `prepare_oscillator_pair_overlap`.

    occupations_osc1 : sequence of int, length = pair_data.num_modes_osc1
        Fock occupations for oscillator 1.

    occupations_osc2 : sequence of int, length = pair_data.num_modes_osc2
        Fock occupations for oscillator 2.

    use_logs : bool, default=False
        Forwarded to `integrate_gaussian_hermite`. Useful when factorial-like
        coefficients become large.

    include_normalization : bool, default=True
        If True, include the standard harmonic-oscillator normalization factors
        for both states. If False, return the raw polynomial-Gaussian integral.

    report_prefactor_as_exponent : bool, default=False
        Forwarded to `integrate_gaussian_hermite`.

    Returns
    -------
    overlap : complex
        The overlap integral if `report_prefactor_as_exponent=False`.

    (coeff, exponent) : tuple[complex, complex]
        If `report_prefactor_as_exponent=True`, returns the coefficient/exponent
        representation from the integration engine, optionally including the
        normalization contribution in the exponent.

    Notes
    -----
    The local wavefunction convention assumed here is

        ψ_{n}(θ)
        = ∏_i [ (ω_i/π)^{1/4} / sqrt(2^{n_i} n_i!) ]
          H_{n_i}(ξ_i) exp[-1/2 ω_i ξ_i^2 / 2? ]

    You should make sure this normalization convention matches the precise
    convention used in your notes/code. In particular, whether the frequency is
    absorbed into the coordinate ξ, or appears explicitly in the Gaussian, must
    be consistent with the meaning of the supplied transforms.
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

    # Harmonic-oscillator normalization factors for the two local states.
    #
    # Adjust this block if your precise convention for the local wavefunctions
    # differs (for example, if frequencies are already absorbed into the rows of S).
    normalization_log = 0.0

    for omega, n in zip(pair_data.freq_osc1, occupations_osc1_arr):
        normalization_log += 0.25 * np.log(omega / np.pi)
        normalization_log -= 0.5 * (n * np.log(2.0) + math.lgamma(n + 1))

    for omega, n in zip(pair_data.freq_osc2, occupations_osc2_arr):
        normalization_log += 0.25 * np.log(omega / np.pi)
        normalization_log -= 0.5 * (n * np.log(2.0) + math.lgamma(n + 1))

    if report_prefactor_as_exponent:
        assert isinstance(integral_result, tuple)
        coeff, exponent = integral_result
        return coeff, exponent + normalization_log

    return integral_result * np.exp(normalization_log)


def batched_overlaps(
    pair_data: OscillatorPairOverlapData,
    occupation_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    use_logs: bool = False,
    include_normalization: bool = True,
) -> list[complex | tuple[complex, complex]]:
    """
    Evaluate a list of overlaps for many occupation pairs using the same fixed
    oscillator pair geometry.
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
