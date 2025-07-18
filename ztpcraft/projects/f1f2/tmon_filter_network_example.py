import numpy as np
import sympy as sm
from scipy.constants import hbar

from typing import Callable

def three_resonator_capacitive_coupling_secular_equation(
    f_01: float,
    f_02: float,
    f_0q: float,
    kappa: float,
    g_12: float,
    g_t1: float,
    g_t2: float,
) -> sm.Expr:
    """
    construct the secular equation to be solved for the eigenmode frequencies of a three-mode system
    where each mode is coupled to the other two modes via charge coupling. The frequencies and kappa
    can be in radian units or frequency units, as long as they are consistent. For example, if
    f_01 and f_02 are in radian units, then kappa is the inverse of the photon lifetime. The lossy port
    is defined as the port that is coupled to the filter mode 2.

    Parameters
    ----------
    f_01 : float
        frequency of the first filter mode
    f_02 : float
        frequency of the second filter mode
    f_0q : float
        frequency of the qubit mode
    kappa : float
        2x the damping rate of the second filter mode (writing the 2nd bare mode frequency as
        f_02 + i*kappa/2)
    g_12 : float
        charge coupling coefficient between the two filters; defined as g_12 = C_12/sqrt(C_1*C_2), where
        C_12 is the coupling capacitance, and C_1 and C_2 are the self-capacitances of the two filters
    g_t1 : float
        charge coupling coefficient between the first filter and the qubit
    g_t2 : float
        charge coupling coefficient between the second filter and the qubit

    Returns
    -------
    polynomial : sympy expression
        the polynomial to be solved for the eigenmode frequencies
    """
    x = sm.symbols("x")
    C_ratio = np.sqrt(g_t1 * g_t2 * g_12)
    return (
        (-(x**2) + f_01**2) * (-(x**2) + f_02**2 + 1j * kappa * x) * (-(x**2) + f_0q**2)
        + 2 * x**6 * C_ratio
        - (-(x**2) + f_01**2) * x**4 * g_t2
        - (-(x**2) + f_02**2 + 1j * kappa * x) * x**4 * g_t1
        - (-(x**2) + f_0q**2) * x**4 * g_12
    )


def mode_freqs_from_secular_equation(
    *args: tuple, secular_equation: Callable
) -> np.ndarray:
    """
    obtain the complex frequencies of the system given the parameters

    Parameters
    ----------
    *args : tuple
        tuple of parameters to be used in the secular equation
    secular_equation : Callable
        the secular equation to be solved for the mode frequencies

    Returns
    -------
    sorted_roots : array
        array of complex roots of the polynomial sorted in ascending order of real
        parts
    """
    # pass through the arguments to the secular equation
    poly = secular_equation(*args)
    root_list = list(sm.roots(poly).keys())
    # obtain the real part of the roots, and get the order of the roots in ascending order
    real_parts = [complex(root).real for root in root_list]
    sorted_indices = np.argsort(real_parts)
    sorted_roots = np.array(root_list)[sorted_indices]
    # convert sympy complex to numpy complex
    sorted_roots = np.array([complex(root) for root in sorted_roots])[3:]
    return sorted_roots


def three_resonator_dressed_basis_langevin_matrix(
    omega_1: float, omega_2: float, omega_3: float, c1: float, c2: float, c3: float
) -> np.ndarray:
    """
    construct the matrix for the Langevin equation for the three-mode system in the dressed basis.
    The matrix M is defined as
    d a_i/dt = sum_j M_ij a_j
    where a_i is the annihilation operator for the i-th mode.
    The setting is that the dissipator takes the form (at zero temperature) D[sum_i c_i * a_i]

    Parameters
    ----------
    omega_1 : float
        frequency of the first mode in angular units
    omega_2 : float
        frequency of the second mode in angular units
    omega_3 : float
        frequency of the third mode in angular units
    c1 : float
        the coefficient of the first mode in the dissipator
    c2 : float
        the coefficient of the second mode in the dissipator
    c3 : float
        the coefficient of the third mode in the dissipator

    Returns
    -------
    M : np.ndarray
        the matrix for the Langevin equation
    """
    M = np.array(
        [
            [
                -1j * omega_1 - 0.5 * c1 * np.conj(c1),
                -0.5 * c1 * np.conj(c2),
                -0.5 * c1 * np.conj(c3),
            ],
            [
                -0.5 * c2 * np.conj(c1),
                -1j * omega_2 - 0.5 * c2 * np.conj(c2),
                -0.5 * c2 * np.conj(c3),
            ],
            [
                -0.5 * c3 * np.conj(c1),
                -0.5 * c3 * np.conj(c2),
                -1j * omega_3 - 0.5 * c3 * np.conj(c3),
            ],
        ],
        dtype=complex,
    )
    return M


def mode_freqs_from_LME(
    omega_1: float, omega_2: float, omega_3: float, c1: float, c2: float, c3: float
) -> tuple:
    """
    obtain the complex frequencies of the system given the parameters

    Parameters
    ----------
    omega_1 : float
        frequency of the first mode in angular units
    omega_2 : float
        frequency of the second mode in angular units
    omega_3 : float
        frequency of the third mode in angular units
    c1 : float
        the coefficient of the first mode in the dissipator
    c2 : float
        the coefficient of the second mode in the dissipator
    c3 : float
        the coefficient of the third mode in the dissipator

    Returns
    -------
    sorted_eigvals : tuple
        tuple of the complex frequencies of the three modes in ascending order of imaginary part
    """
    M = three_resonator_dressed_basis_langevin_matrix(
        omega_1, omega_2, omega_3, c1, c2, c3
    )
    eigvals = np.linalg.eigvals(-M)
    # the imaginary part is the frequency, real part is 1/2 kappa
    # sort them in ascending order of imaginary part
    sorted_indices = np.argsort(eigvals.imag)
    sorted_eigvals = eigvals[sorted_indices]
    return (
        sorted_eigvals[0].imag + 1j * sorted_eigvals[0].real,
        sorted_eigvals[1].imag + 1j * sorted_eigvals[1].real,
        sorted_eigvals[2].imag + 1j * sorted_eigvals[2].real,
    )


def squared_energy_differences(
    x: tuple, func_for_mode_freqs: Callable, target_complex_freqs: tuple
) -> float:
    """
    calculate the squared energy differences between the obtained complex frequencies
    and the target complex frequencies. Currently only implemented for three-mode systems.
    The objective function is defined as the sum of relative errors in the real and imaginary parts
    of the complex frequencies, excluding the imaginary part of the qubit mode.

    Parameters
    ----------
    x : tuple
        tuple of parameters to be used for determining frequencies
    func_for_mode_freqs : Callable
        function for obtaining the complex frequencies of the system
    target_complex_freqs : tuple
        tuple of target complex frequencies

    Returns
    -------
    squared energy differences : float
        sum of the squared energy differences between the obtained complex frequencies
        and the target complex frequencies
    """
    # pass through the arguments to the secular equation
    computed_complex_freqs = func_for_mode_freqs(*x)
    # compute the squared energy differences
    f1, f2, f3 = computed_complex_freqs
    target_f1, target_f2, target_f3 = target_complex_freqs
    return (
        ((f1.real - target_f1.real) / target_f1.real) ** 2
        + ((f2.real - target_f2.real) / target_f2.real) ** 2
        + ((f3.real - target_f3.real) / target_f3.real) ** 2
        + ((f1.imag - target_f1.imag) / target_f1.imag) ** 2
        + ((f2.imag - target_f2.imag) / target_f2.imag) ** 2
    )


def bare_basis_eom(
    y: np.ndarray,
    t: float,
    omega_tuple: tuple,
    g_tuple: tuple,
    kappa: float,
    F: float,
    omega_d: float,
) -> np.ndarray:
    """
    define the equations of motion for the bare basis of a three-mode system with capacitive coupling.
    The equations of motion are defined in the form of a second order differential equation.

    Parameters
    ----------
    y : np.ndarray
        array of the variables to be solved for, in the order of x1, x2, x3, dx1dt, dx2dt, dx3dt
    t : float
        time
    omega_tuple : tuple
        tuple of the frequencies of the three modes in angular units
    g_tuple : tuple
        tuple of the coupling coefficients between the three modes, in the order of g_12_1, g_12_2, g_t1_t, g_t1_1, g_t2_t, g_t2_2
        gij_k is computed by evaluating Cij/Ck. Ck is the self-capacitance of the k-th mode, and Cij is the coupling capacitance
        between the i-th and j-th modes.
    kappa : float
        Inverse of photon lifetime of the second filter mode
    F : float
        drive strength in angular units
    omega_d : float
        drive frequency in angular units
    """
    x1, x2, x3, dx1dt, dx2dt, dx3dt = y
    omega_01, omega_02, omega_0q = omega_tuple
    g12_1, g12_2, gt1_t, gt1_1, gt2_t, gt2_2 = g_tuple
    d2xdt2_before_inv = np.array(
        [
            -(omega_01**2) * x1,
            -(omega_02**2) * x2 - kappa * dx2dt + F * np.sin(omega_d * t),
            -(omega_0q**2) * x3,
        ]
    )
    M_matrix = np.array(
        [
            [1, -g12_1, -gt1_1],
            [-g12_2, 1, -gt2_2],
            [-gt1_t, -gt2_t, 1],
        ]
    )
    d2xdt2 = np.linalg.inv(M_matrix) @ d2xdt2_before_inv
    dydt = np.array([dx1dt, dx2dt, dx3dt, d2xdt2[0], d2xdt2[1], d2xdt2[2]])
    return dydt


def dressed_basis_eom(
    y: np.ndarray,
    t: float,
    omega_tuple: tuple,
    c_tuple: tuple,
    eps: float,
    omega_d: float,
) -> np.ndarray:
    """
    define the equations of motion for the dressed basis of a three-mode system with capacitive coupling.
    The equations of motion are defined in the form of a second order differential equation.

    Parameters
    ----------
    y : np.ndarray
        array of the variables to be solved for, in the order of a1, a2, aq
    t : float
        time
    omega_tuple : tuple
        tuple of the frequencies of the three modes in angular units
    c_tuple : tuple
        tuple of the coefficients of the three modes in the dissipator
    eps : float
        drive strength in angular units
    omega_d : float
        drive frequency in angular units

    Returns
    -------
    dydt : np.ndarray
        array of the time derivatives of the variables
    """
    a1, a2, aq = y
    omega_01, omega_02, omega_0q = omega_tuple
    c1, c2, c3 = c_tuple
    dydt = np.array(
        [
            [
                -1j * (omega_01) - 0.5 * c1 * np.conj(c1),
                -0.5 * c1 * np.conj(c2),
                -0.5 * c1 * np.conj(c3),
            ],
            [
                -0.5 * c2 * np.conj(c1),
                -1j * (omega_02) - 0.5 * c2 * np.conj(c2),
                -0.5 * c2 * np.conj(c3),
            ],
            [
                -0.5 * c3 * np.conj(c1),
                -0.5 * c3 * np.conj(c2),
                -1j * (omega_0q) - 0.5 * c3 * np.conj(c3),
            ],
        ],
        dtype=complex,
    ) @ np.array([a1, a2, aq]) + np.array([c1, c2, c3]) * eps * np.cos(1j * omega_d * t)
    return dydt


def _get_drive_amplitude(
    P: float,
    omega_d: float,
    omega1_opt: float,
    omega2_opt: float,
    omega3_opt: float,
    c1_opt: float,
    c2_opt: float,
    c3_opt: float,
) -> tuple:
    """
    Has not been verified yet.
    Extract the drive amplitude for the three-mode system.
    """

    def heisenberg_LHS(x):
        omega_d = x[0]
        omega_1 = x[1]
        omega_2 = x[2]
        omega_3 = x[3]
        c1 = x[4]
        c2 = x[5]
        c3 = x[6]
        M = np.array(
            [
                [
                    -1j * (omega_1 - omega_d) - 0.5 * c1 * np.conj(c1),
                    -0.5 * c1 * np.conj(c2),
                    -0.5 * c1 * np.conj(c3),
                ],
                [
                    -0.5 * c2 * np.conj(c1),
                    -1j * (omega_2 - omega_d) - 0.5 * c2 * np.conj(c2),
                    -0.5 * c2 * np.conj(c3),
                ],
                [
                    -0.5 * c3 * np.conj(c1),
                    -0.5 * c3 * np.conj(c2),
                    -1j * (omega_3 - omega_d) - 0.5 * c3 * np.conj(c3),
                ],
            ],
            dtype=complex,
        )
        return M

    A_mat = -heisenberg_LHS(
        [omega_d, omega1_opt, omega2_opt, omega3_opt, c1_opt, c2_opt, c3_opt]
    )
    inv_A_mat = np.linalg.inv(A_mat)
    c_vec = np.array([c1_opt, c2_opt, c3_opt])

    B_vec = (1 - c_vec.conj().T @ inv_A_mat @ c_vec) * (-1 / c_vec[0]) * A_mat[0, :]

    g_a = (
        np.sqrt(
            P
            / (hbar * omega_d)
            / (
                (np.abs(B_vec[0] * inv_A_mat[0, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[1] * inv_A_mat[1, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[2] * inv_A_mat[2, :] @ c_vec / 2)) ** 2
            )
        )
        / 1e9
        * c_vec[0]
        / 2
        / np.pi
    )
    g_b = (
        np.sqrt(
            P
            / (hbar * omega_d)
            / (
                (np.abs(B_vec[0] * inv_A_mat[0, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[1] * inv_A_mat[1, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[2] * inv_A_mat[2, :] @ c_vec / 2)) ** 2
            )
        )
        / 1e9
        * c_vec[1]
        / 2
        / np.pi
    )
    g_q = (
        np.sqrt(
            P
            / (hbar * omega_d)
            / (
                (np.abs(B_vec[0] * inv_A_mat[0, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[1] * inv_A_mat[1, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[2] * inv_A_mat[2, :] @ c_vec / 2)) ** 2
            )
        )
        / 1e9
        * c_vec[2]
        / 2
        / np.pi
    )
    return g_a, g_b, g_q
