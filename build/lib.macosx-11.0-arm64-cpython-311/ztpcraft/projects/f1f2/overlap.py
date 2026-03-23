import numpy as np
from scipy.constants import hbar, h
from numpy.typing import ArrayLike


def g_from_overlap_lossless(
    omega_d: ArrayLike,
    overlap_int: ArrayLike,
    omega_mode: ArrayLike,
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
    omega_d : ArrayLike, shape (n_drive,)
        The frequency of the drive in angular frequency (in unit of rad/s).
    overlap_int : ArrayLike, shape (n_drive, n_mode)
        The electric field overlap integral (normalized).
    omega_mode : ArrayLike, shape (n_mode,)
        The frequency of the mode in angular frequency (in unit of rad/s).

    Returns
    -------
    g : ArrayLike, shape (n_drive, n_mode)
        The charge drive strength in Hz, for each drive and each mode.
    """
    omega_d = np.array(omega_d)
    overlap_int = np.array(overlap_int)
    omega_mode = np.array(omega_mode)
    g = np.zeros_like(overlap_int)
    for drive_idx in range(np.size(omega_d)):
        for mode_idx in range(np.size(omega_mode)):
            g[drive_idx, mode_idx] = (
                (
                    (omega_d[drive_idx] ** 2 - omega_mode[mode_idx] ** 2)
                    / omega_mode[mode_idx] ** 2
                )
                * overlap_int[drive_idx, mode_idx]
                / h
            )
    return g


def g_from_overlap_lossy_Kevin(
    omega_d: ArrayLike,
    overlap_int: ArrayLike,
    overlap_phase: ArrayLike,
    omega_mode: ArrayLike,
    kappa_q: ArrayLike,
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
    omega_d : ArrayLike, shape (n_drive,)
        The frequency of the drive in angular frequency (in unit of rad/s).
    overlap_int : ArrayLike, shape (n_drive, n_mode)
        The electric field overlap integral (normalized).
    overlap_phase : ArrayLike, shape (n_drive, n_mode)
        The phase of the overlap integral.
    omega_mode : ArrayLike, shape (n_mode,)
        The frequency of the mode in angular frequency (in unit of rad/s).
    kappa_q : ArrayLike, shape (n_mode,)
        The damping rate of the qubit, 2*the imaginary (angular) qubit frequency.

    Returns
    -------
    g : ArrayLike, shape (n_drive, n_mode)
        The charge drive strength in Hz, for each drive and each mode.
    """
    omega_d = np.array(omega_d)
    overlap_int = np.array(overlap_int)
    omega_mode = np.array(omega_mode)
    kappa_q = np.array(kappa_q)
    g = np.zeros_like(overlap_int) * 1j
    for drive_idx in range(np.size(omega_d)):
        for mode_idx in range(np.size(omega_mode)):
            g[drive_idx, mode_idx] = (
                (
                    (
                        omega_d[drive_idx] ** 2
                        - omega_mode[mode_idx] ** 2
                        + 1j * omega_d[drive_idx] * kappa_q[mode_idx]
                    )
                    / (
                        omega_mode[mode_idx] ** 2
                        - 1j * omega_d[drive_idx] * kappa_q[mode_idx]
                    )
                )
                * overlap_int[drive_idx, mode_idx]
                * np.exp(1j * overlap_phase[drive_idx, mode_idx])
                / h
            )
    return g


def g_from_overlap_lossy_Yao(
    omega_d: ArrayLike,
    overlap_int: ArrayLike,
    overlap_phase: ArrayLike,
    omega_mode: ArrayLike,
    kappa_mode: ArrayLike,
) -> ArrayLike:
    """
    Calculate the charge drive strength (in unit of Hz) from the electric field overlap,
    neglecting losses (assume very small loss). Notice that this function assumes that
    the overlap integral already uses normalized displacement field (i.e. the factor
    1/(2*sqrt(hbar*omega_mode/mode_energy)) is already included in the overlap integral).

    Parameters
    ----------
    omega_d : ArrayLike, shape (n_drive,)
        The frequency of the drive in angular frequency (in unit of rad/s).
    overlap_int : ArrayLike, shape (n_drive, n_mode)
        The electric field overlap integral (normalized).
    overlap_phase : ArrayLike, shape (n_drive, n_mode)
        The phase of the overlap integral.
    omega_mode : ArrayLike, shape (n_mode,)
        The frequency of the mode in angular frequency (in unit of rad/s).
    kappa_mode : ArrayLike, shape (n_mode,)
        The damping rate of the mode, 2*the imaginary (angular) mode frequency.

    Returns
    -------
    g : ArrayLike, shape (n_drive, n_mode)
        The charge drive strength in Hz, for each drive and each mode.
    """
    omega_d = np.array(omega_d)
    overlap_int = np.array(overlap_int)
    omega_mode = np.array(omega_mode)
    kappa_mode = np.array(kappa_mode)
    g = np.zeros_like(overlap_int) * 1j
    for drive_idx in range(np.size(omega_d)):
        for mode_idx in range(np.size(omega_mode)):
            g[drive_idx, mode_idx] = (
                (
                    (
                        omega_d[drive_idx] ** 2
                        - omega_mode[mode_idx] ** 2
                        - kappa_mode[mode_idx] ** 2 / 4
                        - 1j * omega_d[drive_idx] * kappa_mode[mode_idx]
                    )
                    / (omega_mode[mode_idx] ** 2 + kappa_mode[mode_idx] ** 2 / 4)
                )
                * overlap_int[drive_idx, mode_idx]
                * np.exp(1j * overlap_phase[drive_idx, mode_idx])
                / h
            )
    return g


def beta(
    eigenmode_freq: ArrayLike, E_mag: ArrayLike, junction_flux: ArrayLike
) -> ArrayLike:
    """
    Calculate the junction flux participation factor (phi_mj in Minev et al. 2021).
    The method is an alternative to the original Minev et al.'s method; this function
    make use of the junction flux/voltage difference when a given mode is being excited
    to compute the inductive energy participation ratio.

    When using this function, notice that there are a few options for E_mag and junction_flux, for example:
    1. Compute the average magnetic field energy for E_mag [for example, by computing the average electric
       field energy for lumped capacitor-free circuit 1/4 * int( E * conj(D) ), then use
       equipartition theorem to get the average magnetic field energy], then compute the average
       junction flux; this junction flux should be either in phase with magnetic field or
       180 degree out of phase with magnetic field. Based on this pick the EPR sign.
    2. Compute 1/2 of an instantaneous E_mag [for example at phase ±90; this can be evaluated from
       1/4 * int(E . D) at phase 0 if there is no lumped capacitor in the circuit], then compute the
       junction flux at the same phase. This can be done by, for example, computing:
       int_DL E . dl / (2*pi*eigenmode_freq) at phase 0 gives the junction flux at phase -90.
    The only hard requirement is that the junction flux should be in phase with magnetic field energy
    evaluated.

    Parameters
    ----------
    eigenmode_freq: ArrayLike, shape (n_eigenmode,)
        The frequency of the eigenmode in Hz.
    E_mag: ArrayLike, shape (n_eigenmode,)
        The magnetic field energy in J.
    junction_flux: ArrayLike, shape (n_eigenmode, n_junction)
        The flux in the junction in unit of Phi_0.

    Returns
    -------
    beta: ArrayLike, shape (n_eigenmode, n_junction)
        The junction flux participation factor.
    """
    eigenmode_freq = np.array(eigenmode_freq)
    E_mag = np.array(E_mag)
    junction_flux = np.array(junction_flux)
    beta = np.zeros(np.shape(junction_flux))
    for eigenmode_idx in range(np.shape(junction_flux)[0]):
        for junction_idx in range(np.shape(junction_flux)[1]):
            beta[eigenmode_idx, junction_idx] = (
                np.sqrt(h * eigenmode_freq[eigenmode_idx] / (2 * E_mag[eigenmode_idx]))
                * junction_flux[eigenmode_idx, junction_idx]
                * np.pi
            )
    return beta
