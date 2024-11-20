import numpy as np
from scipy.constants import hbar, h
from numpy.typing import ArrayLike


def g_from_overlap_lossless(
    f_p: ArrayLike,
    overlap_int: ArrayLike,
    f_q: ArrayLike,
) -> ArrayLike:
    """
    Calculate the charge drive strength (in unit of Hz) from the electric field overlap,
    neglecting losses (assume very small loss). Notice that this function assumes that
    the overlap integral already uses normalized displacement field (i.e. the factor
    sqrt(2*pi*hbar*freq_q/mode_energy) is already included in the overlap integral).
    Also notice that the expression does not have a factor of 2 in the denominator, which
    is "compatible" with HFSS calculation results but is differ from the paper expression.

    Parameters
    ----------
    f_p : ArrayLike, shape (n_drive,)
        The frequency of the drive in Hz.
    overlap_int : ArrayLike, shape (n_drive, n_mode)
        The electric field overlap integral (normalized).
    f_q : float, shape (n_mode,)
        The frequency of the qubit in Hz.

    Returns
    -------
    g : ArrayLike, shape (n_drive, n_mode)
        The charge drive strength in Hz, for each drive and each mode.
    """
    f_p = np.array(f_p)
    overlap_int = np.array(overlap_int)
    f_q = np.array(f_q)
    g = np.zeros_like(overlap_int)
    for drive_idx in range(np.size(f_p)):
        for mode_idx in range(np.size(f_q)):
            g[drive_idx, mode_idx] = (
                ((f_p[drive_idx] ** 2 - f_q[mode_idx] ** 2) / f_q[mode_idx] ** 2)
                * overlap_int[drive_idx, mode_idx]
                / h
            )
    return g


def g_from_overlap_lossy_Kevin(
    f_p: ArrayLike,
    overlap_int: ArrayLike,
    overlap_phase: ArrayLike,
    f_q: ArrayLike,
    gamma_q: ArrayLike,
) -> ArrayLike:
    """
    Calculate the charge drive strength (in unit of Hz) from the electric field overlap,
    neglecting losses (assume very small loss). Notice that this function assumes that
    the overlap integral already uses normalized displacement field (i.e. the factor
    sqrt(2*pi*hbar*freq_q/mode_energy) is already included in the overlap integral).
    Also notice that the expression does not have a factor of 2 in the denominator, which
    is "compatible" with HFSS calculation results but is differ from the paper expression.

    Parameters
    ----------
    f_p : ArrayLike, shape (n_drive,)
        The frequency of the drive in Hz.
    overlap_int : ArrayLike, shape (n_drive, n_mode)
        The electric field overlap integral (normalized).
    f_q : float, shape (n_mode,)
        The frequency of the qubit in Hz.

    Returns
    -------
    g : ArrayLike, shape (n_drive, n_mode)
        The charge drive strength in Hz, for each drive and each mode.
    """
    f_p = np.array(f_p)
    overlap_int = np.array(overlap_int)
    f_q = np.array(f_q)
    gamma_q = np.array(gamma_q)
    g = np.zeros_like(overlap_int) * 1j
    for drive_idx in range(np.size(f_p)):
        for mode_idx in range(np.size(f_q)):
            g[drive_idx, mode_idx] = (
                (
                    (
                        f_p[drive_idx] ** 2
                        - f_q[mode_idx] ** 2
                        + 1j * f_p[drive_idx] * gamma_q[mode_idx]
                    )
                    / (f_q[mode_idx] ** 2 - 1j * f_p[drive_idx] * gamma_q[mode_idx])
                )
                * overlap_int[drive_idx, mode_idx]
                * np.exp(1j * overlap_phase[drive_idx, mode_idx])
                / h
            )
    return g


def g_from_overlap_lossy_Yao(
    f_p: ArrayLike,
    overlap_int: ArrayLike,
    overlap_phase: ArrayLike,
    f_q: ArrayLike,
    gamma_q: ArrayLike,
) -> ArrayLike:
    """
    Calculate the charge drive strength (in unit of Hz) from the electric field overlap,
    neglecting losses (assume very small loss). Notice that this function assumes that
    the overlap integral already uses normalized displacement field (i.e. the factor
    sqrt(2*pi*hbar*freq_q/mode_energy) is already included in the overlap integral).
    Also notice that the expression does not have a factor of 2 in the denominator, which
    is "compatible" with HFSS calculation results but is differ from the paper expression.

    Parameters
    ----------
    f_p : ArrayLike, shape (n_drive,)
        The frequency of the drive in Hz.
    overlap_int : ArrayLike, shape (n_drive, n_mode)
        The electric field overlap integral (normalized).
    f_q : float, shape (n_mode,)
        The frequency of the qubit in Hz.

    Returns
    -------
    g : ArrayLike, shape (n_drive, n_mode)
        The charge drive strength in Hz, for each drive and each mode.
    """
    f_p = np.array(f_p)
    overlap_int = np.array(overlap_int)
    f_q = np.array(f_q)
    gamma_q = np.array(gamma_q)
    g = np.zeros_like(overlap_int) * 1j
    for drive_idx in range(np.size(f_p)):
        for mode_idx in range(np.size(f_q)):
            g[drive_idx, mode_idx] = (
                (
                    (
                        f_p[drive_idx] ** 2
                        - f_q[mode_idx] ** 2
                        + 1j * f_p[drive_idx] * gamma_q[mode_idx]
                    )
                    / (f_q[mode_idx] ** 2)
                )
                * overlap_int[drive_idx, mode_idx]
                * np.exp(1j * overlap_phase[drive_idx, mode_idx])
                / h
            )
    return g


def beta(
    eigenmode_freq: ArrayLike, E_elec: ArrayLike, junction_flux: ArrayLike
) -> ArrayLike:
    """
    Parameters
    ----------
    eigenmode_freq: ArrayLike, shape (n_eigenmode,)
        The frequency of the eigenmode in Hz.
    E_elec: ArrayLike, shape (n_eigenmode,)
        The electric field energy in J.
    junction_flux: ArrayLike, shape (n_eigenmode, n_junction)
        The flux in the junction in unit of Phi_0.

    Returns
    -------
    beta: ArrayLike, shape (n_eigenmode, n_junction)
    """
    eigenmode_freq = np.array(eigenmode_freq)
    E_elec = np.array(E_elec)
    junction_flux = np.array(junction_flux)
    beta = np.zeros(np.shape(junction_flux))
    for eigenmode_idx in range(np.shape(junction_flux)[0]):
        for junction_idx in range(np.shape(junction_flux)[1]):
            beta[eigenmode_idx, junction_idx] = (
                np.sqrt(h * eigenmode_freq[eigenmode_idx] / (2 * E_elec[eigenmode_idx]))
                * junction_flux[eigenmode_idx, junction_idx]
                * 2
                * np.pi
            )
    return beta
