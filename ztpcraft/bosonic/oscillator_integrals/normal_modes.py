# normal_modes.py

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NormalModeResult:
    frequencies: Array1D
    transform_xi_theta: Array2D
    transform_qn: Array2D


def diagonalize_quadratic_hamiltonian(
    EC: Array2D,
    Gamma: Array2D,
) -> NormalModeResult:
    """
    Perform diagonalization:

        8 EC^{1/2} Γ EC^{1/2} = U E^2 U^T
    """

    # diagonalize EC
    evals, evecs = np.linalg.eigh(EC)

    EC_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    EC_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    M = 8.0 * EC_sqrt @ Gamma @ EC_sqrt

    eigvals, U = np.linalg.eigh(M)

    frequencies = np.sqrt(eigvals)

    S_xi_theta = U.T @ EC_inv_sqrt / np.sqrt(8.0)
    S_qn = U.T @ EC_sqrt * np.sqrt(8.0)

    return NormalModeResult(
        frequencies=frequencies,
        transform_xi_theta=S_xi_theta,
        transform_qn=S_qn,
    )
