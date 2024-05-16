import numpy as np
from scipy.constants import hbar
from numpy.typing import ArrayLike


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
    f_q: float, g: ArrayLike, matelem: ArrayLike, resistance: float, source_power: float
) -> float:
    g = np.array(g)
    matelem = np.array(matelem)
    chi = g / voltage_max_yao(source_power, resistance)
    gamma = (2 * f_q * 2 * np.pi * resistance / hbar) * np.abs(
        np.sum(chi * matelem)
    ) ** 2
    return gamma
