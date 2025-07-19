import numpy as np

from numpy.typing import NDArray

from scipy.special import factorial


def n_bose_freq_to_kbt(n_bose: float, freq: float) -> float:
    """
    frequency in GHz, n_bose in number of phonons
    return kbt in terms of GHz
    """
    return freq / np.log(1 / n_bose + 1)


def kbt_freq_to_n_bose(kbt: float, freq: float) -> float:
    """
    frequency in GHz, kbt in terms of GHz
    return n_bose in number of phonons
    """
    return 1 / (np.exp(freq / kbt) - 1)


def thermal_state_diag(evals: NDArray[np.float64], kbt: float) -> NDArray[np.float64]:
    """
    evals in GHz, kbt in terms of GHz
    return the diagonal element of the density matrix of a thermal state
    """
    return np.exp(-evals / kbt) / np.sum(np.exp(-evals / kbt))


def poisson(n: int, nbar: float) -> float:
    """
    n is the number of phonons, nbar is the average number of phonons
    return the Poisson distribution
    """
    return np.power(nbar, n) * np.exp(-nbar) / factorial(n)
