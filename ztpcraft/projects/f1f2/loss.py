import numpy as np
from scipy.constants import hbar, Boltzmann, h
from numpy.typing import ArrayLike
from typing import Callable


def voltage_max_yao(power: float, resistance: float) -> float:
    """
    Expression from Yao, this prefactor 8 is mysterious to me.

    Parameters
    ----------
    power : float
        The power in W, used in HFSS.
    resistance : float
        The resistance in Ohm.
    """
    return np.sqrt(8 * power * resistance)


def decay_rate(
    f_q: float,
    g: ArrayLike,
    matelem: ArrayLike,
    resistance: float,
    source_power: float,
    S_V: Callable,
) -> float:
    """
    General expression for decay rate.

    Parameters
    ----------
    f_q : float
        The qubit frequency in Hz.
    g : ArrayLike
        The coupling strength between qubit and cavity.
    matelem : ArrayLike
        The matrix element of the qubit operator. The unit of g*matelem is energy.
    resistance : float
        The resistance of the port in Ohm.
    source_power : float
        The power in W, used in the simulation.
    S_V : Callable
        The voltage noise spectral density in V^2/Hz, should be a function of frequency in Hz, i.e. S_V(f).

    Returns
    -------
    The decay rate in Hz
    """
    g = np.array(g)
    matelem = np.array(matelem)
    chi = g / voltage_max_yao(source_power, resistance)
    gamma = (1 / hbar**2) * np.abs(np.sum(chi * matelem)) ** 2 * S_V(f_q)
    return gamma


def S_quantum_johnson_nyquist(f: float, T: float, Z_0: float) -> float:
    """
    The quantum Johnson-Nyquist noise spectral density.

    Parameters
    ----------
    f : float
        The frequency in Hz.
    T : float
        The temperature in K.
    Z_0 : float
        The impedance in Ohm.

    Returns
    -------
    The spectral density in V^2/Hz
    """
    return 2 * h * f * Z_0 * (1 + Bose_factor(f, T))


def Bose_factor(f, T):
    """
    The Bose factor.

    Parameters
    ----------
    f : float
        The frequency in Hz.
    T : float
        The temperature in K.

    Returns
    -------
    The Bose factor.
    """
    return 1 / (np.exp(h * f / (Boltzmann * T)) - 1)
