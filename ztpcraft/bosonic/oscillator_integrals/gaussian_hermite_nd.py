import math
from typing import Any, Literal, overload
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt

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
        if np.isclose(coeff, 0.0, atol=1e-14, rtol=0.0):
            return complex(0.0, 0.0)
        else:
            log_value: complex = np.log(coeff) + prefactor_exponent + Z0_exponent
            return np.exp(log_value)