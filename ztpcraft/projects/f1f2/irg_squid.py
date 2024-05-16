import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Tuple, Optional
from numpy.typing import ArrayLike


def LJ1_LJ2_2_L_inv(LJ1: float, LJ2: float) -> Tuple[float, float, float]:
    """
    Compute the inverse of the Josephson inductance of the first junction, the
    second junction, and the sum of the two junctions.

    Parameters
    ----------
    LJ1 : float
        The Josephson inductance of the first junction.
    LJ2 : float
        The Josephson inductance of the second junction.

    Returns
    -------
    The inverse of the Josephson inductance of the first junction, the second
    junction, and the sum of the two junctions.
    """
    LJ1_inv = 1 / LJ1
    LJ2_inv = 1 / LJ2
    LSigma_inv = LJ1_inv + LJ2_inv
    return LJ1_inv, LJ2_inv, LSigma_inv


def LJ1_LJ2_fq_2_C(LJ1, LJ2, fq):
    """
    LJ1, LJ2 in nH, fq in GHz, C in nF
    """
    LJ1_inv, LJ2_inv, LSigma_inv = LJ1_LJ2_2_L_inv(LJ1, LJ2)
    omega_q = 2 * np.pi * fq  # GHz -> 1E9 rad/s
    C = 1 / omega_q**2 * LSigma_inv  # 1/omega_q^2 = L*C
    return C


def LJ1_LJ2_C_2_fq(LJ1, LJ2, C):
    LJ1_inv, LJ2_inv, LSigma_inv = LJ1_LJ2_2_L_inv(LJ1, LJ2)
    fq = 1 / np.sqrt(C / LSigma_inv) / (2 * np.pi)
    return fq


def f1f2_SQUID_off_resonance(
    A1: ArrayLike,
    A2: ArrayLike,
    f_p: ArrayLike,
    LJ1: float,
    LJ2: float,
    f_q: float,
    theta_s_1: Optional[ArrayLike] = None,
    theta_s_2: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the f1 and f2 of a SQUID off resonance.

    Parameters
    ----------

    """
    # check A1 and A2: if they are float or array
    if isinstance(A1, (int, float)):
        A1 = np.array([A1])
        A2 = np.array([A2])
        f_p = np.array([f_p])
    else:
        A1 = np.array(A1)
        A2 = np.array(A2)
        f_p = np.array(f_p)
    A1_sign = np.ones_like(A1)
    LJ1_inv, LJ2_inv, LSigma_inv = LJ1_LJ2_2_L_inv(LJ1, LJ2)
    # if the phase difference is not given, assume A1 and A2 have the same sign
    if theta_s_1 is None and theta_s_2 is None:
        A2_sign = np.ones_like(A1)
    # if the phase difference is given, compute the sign of A2 from phase
    elif theta_s_1 is not None and theta_s_2 is not None:
        theta_s_1 = np.array(theta_s_1)
        theta_s_2 = np.array(theta_s_2)
        # for each entry of theta_s_1 and theta_s_2, compute
        A2_sign = np.where(
            np.isclose(np.abs(theta_s_1 - theta_s_2), 0, atol=0.1), 1, -1
        )
    f1_method_off_resonance = A1 * A1_sign - f_q**2 / LSigma_inv * np.power(
        f_p, -2
    ) * (LJ1_inv * A1 * A1_sign + LJ2_inv * A2 * A2_sign)
    f2_method_off_resonance = A2 * A2_sign - f_q**2 / LSigma_inv * np.power(
        f_p, -2
    ) * (LJ1_inv * A1 * A1_sign + LJ2_inv * A2 * A2_sign)
    return f1_method_off_resonance, f2_method_off_resonance


def g_from_overlap(
    A1: ArrayLike,
    A2: ArrayLike,
    omega_p: ArrayLike,
    LJ1: float,
    LJ2: float,
    omega_q: float,
    theta_s_1: Optional[ArrayLike] = None,
    theta_s_2: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the f1 and f2 of a SQUID off resonance.

    Parameters
    ----------

    """
    # check A1 and A2: if they are float or array
    if isinstance(A1, (int, float)):
        A1 = np.array([A1])
        A2 = np.array([A2])
        omega_p = np.array([omega_p])
    else:
        A1 = np.array(A1)
        A2 = np.array(A2)
        omega_p = np.array(omega_p)
    A1_sign = np.ones_like(A1)
    LJ1_inv, LJ2_inv, LSigma_inv = LJ1_LJ2_2_L_inv(LJ1, LJ2)
    # if the phase difference is not given, assume A1 and A2 have the same sign
    if theta_s_1 is None and theta_s_2 is None:
        A2_sign = np.ones_like(A1)
    # if the phase difference is given, compute the sign of A2 from phase
    elif theta_s_1 is not None and theta_s_2 is not None:
        theta_s_1 = np.array(theta_s_1)
        theta_s_2 = np.array(theta_s_2)
        # for each entry of theta_s_1 and theta_s_2, compute
        A2_sign = np.where(
            np.isclose(np.abs(theta_s_1 - theta_s_2), 0, atol=0.1), 1, -1
        )
    f1_method_off_resonance = A1 * A1_sign - omega_q**2 / LSigma_inv * np.power(
        omega_p, -2
    ) * (LJ1_inv * A1 * A1_sign + LJ2_inv * A2 * A2_sign)
    f2_method_off_resonance = A2 * A2_sign - omega_q**2 / LSigma_inv * np.power(
        omega_p, -2
    ) * (LJ1_inv * A1 * A1_sign + LJ2_inv * A2 * A2_sign)
    return f1_method_off_resonance, f2_method_off_resonance


def f1f2_SQUID_off_resonance_optimize(
    A1: ArrayLike,
    A2: ArrayLike,
    f_p: ArrayLike,
    LJ1: float,
    LJ2: float,
    f_q_initial_guess: float,
    theta_s_1: Optional[ArrayLike] = None,
    theta_s_2: Optional[ArrayLike] = None,
) -> OptimizeResult:
    args = (
        A1,
        A2,
        theta_s_1,
        theta_s_2,
        LJ1,
        LJ2,
        f_p,
    )

    def cost_function(f_q, *args):
        A1, A2, theta_s_1, theta_s_2, LJ1, LJ2, f_p = args
        f1_list, f2_list = f1f2_SQUID_off_resonance(
            A1=A1,
            A2=A2,
            f_p=f_p,
            LJ1=LJ1,
            LJ2=LJ2,
            f_q=f_q,
            theta_s_1=theta_s_1,
            theta_s_2=theta_s_2,
        )
        f1_list = np.array(f1_list)
        f2_list = np.array(f2_list)
        f1_mean = np.mean(f1_list)
        f2_mean = np.mean(f2_list)
        err = np.sum((f1_list - f1_mean) ** 2) + np.sum((f2_list - f2_mean) ** 2)
        return err

    sol = minimize(cost_function, f_q_initial_guess, args=args, method="Nelder-Mead")
    return sol
