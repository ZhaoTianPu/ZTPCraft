import numpy as np
from scipy.constants import hbar, Boltzmann, h
from numpy.typing import ArrayLike
from typing import Callable, List
from scqubits import GenericQubit


def peak_voltage_impedance_matched(power: float, resistance: float) -> float:
    """
    The peak voltage of a port with impedance matched. The impedance matching condition is Z_port = Z_load*.
    Reference: Pozar Microwave Engineering, Ch. 4.

    In HFSS, the lumped port seem to assume perfectly matched impedance.

    Parameters
    ----------
    power : float
        The power in W.
    resistance : float
        The resistance in Ohm.
    """
    return np.sqrt(8 * power * resistance)


def decay_rate(
    f_q: float,
    suscep: ArrayLike | List,
    matelem: ArrayLike | List,
    S_V: Callable,
) -> float:
    """
    General expression for decay rate.

    Parameters
    ----------
    f_q : float
        The qubit frequency in Hz.
    suscep : ArrayLike
        The susceptibility of the qubit drive parameter g wrt the input voltage. Unit of g is defined such that
        g * matelem has dimension of energy. Can be a number or a function of frequency in Hz, i.e. suscep(f).
    matelem : ArrayLike
        The matrix element of the qubit operator. The unit of g*matelem is energy.
    resistance : float
        The resistance of the port in Ohm.
    S_V : Callable
        The voltage noise spectral density in V^2/Hz, should be a function of frequency in Hz, i.e. S_V(f).

    Returns
    -------
    The decay rate in Hz
    """
    matelem = np.array(matelem)
    suscep = np.array(suscep)
    suscep_matelem_prod_total = 0
    for term_idx in range(len(suscep)):
        if isinstance(suscep[term_idx], Callable):
            suscep_matelem_prod_total += (
                suscep[term_idx](np.abs(f_q)) * matelem[term_idx]
            )
        else:
            suscep_matelem_prod_total += suscep[term_idx] * matelem[term_idx]
    gamma = (1 / hbar**2) * np.abs(suscep_matelem_prod_total) ** 2 * S_V(f_q)
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


def t1_flux_line_impedance(
    i: int,
    j: int,
    qubit: "GenericQubit",
    Z0: float,
    T: float,
    suscep: List,
    matelem: List,
    total_rate: bool = True,
    get_rate: bool = False,
):
    """
    Calculate the relaxation time T1 due to charge line for a qubit.

    Parameters
    ----------
    i : int
        The index of the initial state.
    j : int
        The index of the final state.
    qubit : GenericQubit
        The qubit object.
    Z0 : float
        The impedance of the port in Ohm.
    T : float
        The temperature in K.
    suscep : ArrayLike | List
        The susceptibility of the qubit drive parameter g wrt the input voltage. Unit of g is defined such that
        g * matelem has dimension of energy. Can be a number or a function of frequency in Hz, i.e. suscep(f).
    matelem : ArrayLike | List
        The matrix element of the qubit operator. The unit of g*matelem is energy.
        Accept following forms of entries, or list of entries:
            - a tuple of (matrix element operator string, prefactor)
            - a tuple of (matrix element table, prefactor)
            - a tuple of (matrix element method, prefactor)
            - a numpy array
    total_rate : bool
        If True, the total rate is calculated. If False, only the rate from i to j is calculated.
    get_rate : bool
        If True, return the rate. If False, return the relaxation time T1.

    Returns
    -------
        The relaxation time T1 in s, or the rate in Hz.
    """
    eigenvals, eigenvecs = qubit.eigensys(evals_count=max(i, j) + 1)

    matelem_list = []

    for matelem_term in matelem:
        if isinstance(matelem_term, tuple):
            if isinstance(matelem_term[0], str):
                matelem_term = (
                    eigenvecs[:, i].conj().T
                    @ getattr(qubit, matelem_term[0])()
                    @ eigenvecs[:, j]
                    * matelem_term[1]
                )
            elif callable(matelem_term[0]):
                matelem_term = (
                    eigenvecs[:, i].conj().T
                    @ matelem_term[0]()
                    @ eigenvecs[:, j]
                    * matelem_term[1]
                )
            else:
                matelem_term = matelem[0][i, j] * matelem_term[1]
        matelem_list.append(matelem_term)

    freq_ij = eigenvals[i] - eigenvals[j]
    rate_ij = decay_rate(
        f_q=freq_ij * 1e9,  # GHz to Hz
        suscep=suscep,
        matelem=matelem_list,
        S_V=lambda f: S_quantum_johnson_nyquist(f, T, Z0),
    )
    rate_final = rate_ij
    if total_rate:
        rate_ji = decay_rate(
            f_q=-freq_ij * 1e9,  # GHz to Hz
            suscep=suscep,
            matelem=matelem_list,
            S_V=lambda f: S_quantum_johnson_nyquist(f, T, Z0),
        )
        rate_final += rate_ji
    if get_rate:
        return rate_final
    return 1 / rate_final
