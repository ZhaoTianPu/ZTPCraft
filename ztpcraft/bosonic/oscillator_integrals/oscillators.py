from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .normal_modes import diagonalize_quadratic_hamiltonian

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LocalHarmonicOscillator:
    """
    Represents a single N-dimensional harmonic oscillator in global coordinates.
    """

    transform_xi_theta: Array2D
    transform_qn: Array2D
    frequencies: Array1D
    center: Array1D

    def quadratic_form(self) -> Array2D:
        """Return K = S^T diag(omega) S."""
        return (
            self.transform_xi_theta.T
            @ np.diag(self.frequencies)
            @ self.transform_xi_theta
        )

    def log_jacobian(self) -> float:
        return np.log(abs(np.linalg.det(self.transform_xi_theta)))

    def q_to_n_matrix(self) -> Array2D:
        return np.asarray(np.linalg.inv(self.transform_qn), dtype=np.float64)

    def n_to_q_matrix(self) -> Array2D:
        return self.transform_qn

    @property
    def num_modes(self) -> int:
        return self.transform_xi_theta.shape[0]

    @property
    def dim(self) -> int:
        return self.transform_xi_theta.shape[1]


def oscillator_from_circuit(
    EC: Array2D,
    Gamma: Array2D,
    theta_center: Array1D,
) -> LocalHarmonicOscillator:
    """
    Construct LocalHarmonicOscillator from circuit matrices.
    """
    nm = diagonalize_quadratic_hamiltonian(EC, Gamma)
    return LocalHarmonicOscillator(
        transform_xi_theta=nm.transform_xi_theta,
        transform_qn=nm.transform_qn,
        frequencies=nm.frequencies,
        center=theta_center,
    )


__all__ = [
    "LocalHarmonicOscillator",
    "oscillator_from_circuit",
]
