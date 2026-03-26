"""Normal-mode diagonalization helpers."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NormalModeResult:
    frequencies: Array1D
    transform_xi_theta: Array2D
    transform_qn: Array2D
    xi_zpf: Array1D
    q_zpf: Array1D
    xi_osc_length: Array1D


def diagonalize_quadratic_hamiltonian(
    EC: Array2D,
    Gamma: Array2D,
    atol: float = 1e-12,
) -> NormalModeResult:
    """
    Diagonalize:
        H = 4 n^T EC n + 1/2 theta^T Gamma theta
    """
    ec_sym = 0.5 * (EC + EC.T)
    gamma_sym = 0.5 * (Gamma + Gamma.T)

    ec_prime = 8.0 * ec_sym

    # eigen-decomposition for sqrt
    evals, evecs = np.linalg.eigh(ec_prime)
    ec_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    ec_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    mode_matrix = ec_sqrt @ gamma_sym @ ec_sqrt
    lam, U = np.linalg.eigh(mode_matrix)

    frequencies = cast(Array1D, np.sqrt(np.clip(lam, 0.0, None)))

    S_xi_theta = cast(Array2D, U.T @ ec_inv_sqrt)
    S_qn = cast(Array2D, U.T @ ec_sqrt)

    with np.errstate(divide="ignore", invalid="ignore"):
        xi_zpf = cast(Array1D, np.sqrt(1 / (2.0 * frequencies)))
        q_zpf = cast(Array1D, np.sqrt(frequencies / 2.0))
        xi_zpf = cast(Array1D, np.where(frequencies > atol, xi_zpf, np.inf))
        q_zpf = cast(Array1D, np.where(frequencies > atol, q_zpf, 0.0))
        xi_osc_length = cast(Array1D, np.sqrt(1 / frequencies))

    return NormalModeResult(
        frequencies=frequencies,
        transform_xi_theta=S_xi_theta,
        transform_qn=S_qn,
        xi_zpf=xi_zpf,
        q_zpf=q_zpf,
        xi_osc_length=xi_osc_length,
    )
