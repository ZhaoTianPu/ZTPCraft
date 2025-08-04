import numpy as np
from scipy.constants import hbar
import scipy as sp

from numpy.typing import NDArray
from typing import Callable, Union, Literal, Optional, Dict, Any

from scipy.integrate._ivp.ivp import OdeResult


def power_kappa_frequency_to_drive_strength(
    input_power: float, kappa: float, omega_r: float
) -> float:
    """
    For a harmonic oscillator, convert input power, kappa, and resonance frequency to drive strength
    when the resonator is driven at the resonance frequency. This expression is based on the standard
    input-output relation, see, e.g. Ganjam et al. Nat. Comm. 15, 1 (2024) Supplementary Note 10.
    The drive takes the form epsilon*sin(omega_d*t + phi), and epsilon is returned.

    The expression is
    g = sqrt(4*P*kappa/(hbar*omega_r))*1E-9/2/pi

    A sketch of the derivation:
    From input-output relation, when the drive is on resonance with the resonator,
    the steady state solution of the mode relates to input mode as:
    sqrt(kappa)*b_in = kappa/2*a_ss
    then from input-output theory,
    input_power = hbar*omega_r*kappa/4*|a_ss|^2
    so kappa*a_ss = sqrt(input_power*4*kappa/hbar/omega_r)
    since we want the drive strength in form of sin/cos,
    we multiply a factor of 2 to sqrt(kappa)*b_in (which represents drive as
    exp(i*omega_d*t + phi))

    Parameters
    ----------
    input_power : float
        Power in watts.
    kappa : float
        Inverse of photon lifetime.
    omega_r : float
        Resonance frequency in 2pi*GHz.
        The complex angular frequency of the resonator is omega_r + i*kappa/2.

    Returns
    -------
    drive_strength : float
        Drive strength in units of GHz*2*pi (compatible with using angular frequency in static Hamiltonian).
        The drive takes the form epsilon*sin(omega_d*t + phi), and epsilon is returned.
    """
    return np.sqrt(4 * input_power * kappa / (hbar * omega_r)) * 1e-9


def power_dissipator_coeff_frequency_to_drive_strength(
    input_power: float, dissipator_coeff: float, omega_d: float
) -> float:
    """
    For a harmonic oscillator, convert input power, dissipator coefficient, and drive frequency to drive strength
    when the resonator is driven at the drive frequency. This expression is based on the standard
    input-output relation.
    The dissipator is defined as D[sum_i c_i * a_i], where a_i is the annihilation operator for the i-th mode.
    The expression is
    g = 2*sqrt(P/(hbar*omega_d))*dissipator_coeff*1E-9

    Parameters
    ----------
    input_power : float
        Power in watts.
    dissipator_coeff : float
        Dissipator coefficient.
    omega_d : float
        Drive frequency in 2pi*GHz.

    Returns
    -------
    drive_strength : float
        Drive strength in units of GHz*2*pi (compatible with using angular frequency in static Hamiltonian).
        The drive takes the form epsilon*sin(omega_d*t + phi), and epsilon is returned.
    """
    return 2 * np.sqrt(input_power / (hbar * omega_d)) * dissipator_coeff * 1e-9


def steady_state_displacement(
    omega_r: float,
    kappa: float,
    epsilon: float,
    omega_d: float,
    drive_phase: float,
    apply_rwa: bool = False,
    return_type: Literal[
        "func_complex",
        "func_real",
        "func_imag",
        "amp_real",
        "amp_imag",
        "phase_real",
        "phase_imag",
    ] = "func_complex",
) -> Union[Callable, float]:
    """
    Calculate the steady-state displacement of a harmonic oscillator driven at a frequency omega_d.
    Without RWA, the Hamiltonian is:
        H = ℏ*ω_r*a†a + i*ℏ*ε*sin(ω_d*t + φ)*(a† - a)

    Taking into account the damping, the quantum Langevin equation is:
        da/dt = -(iω_r + κ/2) a + ε*sin(ω_d*t + φ)

    The steady-state displacement is given by
        a_ss = -epsilon/2*[exp(i(ω_d*t+φ))/(ω_r + ω_d - i*κ/2) - exp(-i(ω_d*t+φ))/(ω_r - ω_d - i*κ/2)]

    In the case of RWA being applied, the Hamiltonian is defined as
        H = ℏ*ω_r*a†a + ℏ*ε/2*exp(i*(ω_d*t+φ))(-a) + h.c.

    The resulting quantum Langevin equation is
        da/dt = -(iω_r + κ/2) a - i*ε/2*exp(-i(ω_d*t+φ))

    The steady-state displacement is given by
        a_ss = epsilon/2*exp(-i(ω_d*t+φ))/(ω_r - ω_d - i*κ/2)

    Parameters
    ----------
    omega_r : float
        Resonance frequency in 2pi*GHz.
    kappa : float
        Damping rate in 2pi*GHz.
    epsilon : float
        Drive strength in units of 2pi*GHz.
    omega_d : float
        Drive frequency in 2pi*GHz.
    drive_phase : float
        Drive phase in radians.
    apply_rwa : bool
        Whether to apply the rotating wave approximation.
    return_type : Literal["func_complex", "func_real", "func_imag", "amp_real", "amp_imag", "phase_real", "phase_imag"]
        The type of return value. For the type "amp_real", "phase_real", amplitude and phase returned are the real part
        of the steady state displacement of form amp*cos(omega_d*t + phase). For the type "amp_imag", "phase_imag",
        amplitude and phase returned are the imaginary part of the steady state displacement of form amp*sin(omega_d*t + phase).
    """
    i = 1j
    denom = omega_r - omega_d - i * kappa / 2
    denom_plus = omega_r + omega_d - i * kappa / 2
    denom_minus = omega_r - omega_d - i * kappa / 2

    def displacement_func(t_val: float) -> complex:
        phase = omega_d * t_val + drive_phase
        exp_plus = np.exp(i * phase)
        exp_minus = np.exp(-i * phase)

        if apply_rwa:
            return (epsilon / 2) * exp_minus / denom
        else:
            return -(epsilon / 2) * (exp_plus / denom_plus - exp_minus / denom_minus)

    # Dispatch according to return_type
    if return_type == "func_complex":
        return displacement_func
    elif return_type == "func_real":
        return lambda t_val: np.real(displacement_func(t_val))
    elif return_type == "func_imag":
        return lambda t_val: np.imag(displacement_func(t_val))
    elif return_type == "amp_real":
        if apply_rwa:
            return abs((epsilon / 2) / denom)
        else:
            amplitude = (
                epsilon / 2 * np.abs(1 / denom_plus - 1 / denom_minus.conjugate())
            )
            return amplitude
    elif return_type == "phase_real":
        if apply_rwa:
            return drive_phase
        else:
            phase = np.angle(1 / denom_plus - 1 / denom_minus.conjugate())
            return float(phase + drive_phase + np.pi)
    elif return_type == "amp_imag":
        if apply_rwa:
            return abs((epsilon / 2) / denom)
        else:
            amplitude = (
                epsilon / 2 * np.abs(1 / denom_plus + 1 / denom_minus.conjugate())
            )
            return amplitude
    elif return_type == "phase_imag":
        if apply_rwa:
            return drive_phase
        else:
            phase = np.angle(
                1 / denom_plus + 1 / denom_minus.conjugate()
            )  # check this phase
            return float(phase + drive_phase)
    else:
        raise ValueError(f"Invalid return_type: {return_type}")


def envelope_func(
    t: float,
    ramp_time_const: float,
    ramp_type: Literal["linear", "gaussian", "tanh", "sin"] = "linear",
    single_sided: bool = True,
    flat_top_time: Optional[float] = None,
) -> float:
    """
    Generate an envelope function for a ramp pulse.

    Parameters
    ----------
    t : float
        Time in seconds.
    ramp_time_const : float
        Ramp time constant in seconds.
    ramp_type : Literal["linear", "gaussian", "tanh", "sin"]
        Ramp type.
    single_sided : bool
        Whether the ramp is single-sided.
    flat_top_time : Optional[float]
        Flat top time in seconds.

    Returns
    -------
    envelope : float
        Value of the envelope function at time t.
    """
    if ramp_type != "linear" or not single_sided:
        raise NotImplementedError("Only single-sidedlinear ramp is implemented for now")
    else:
        if t < ramp_time_const:
            # linear ramp
            raw_envelope = t / (ramp_time_const)
        else:
            raw_envelope = 1
        return raw_envelope


def enveloped_cos(
    t: float,
    omega_d: float,
    t_ramp: float,
    envelope_func: Callable[[Any], float],
    envelope_func_args: Dict[str, float] = {},
) -> float:
    return envelope_func(t, t_ramp, **envelope_func_args) * np.cos(omega_d * t)


def solve_oscillator_under_drive(
    t_sample: NDArray[np.float64],
    t_ramp: float,
    drive_strength: float,
    omega_q: float,
    omega_d: float,
    init_displacement: complex = 0j,
    envelope_func_args: Dict[str, float] = {},
) -> "OdeResult":
    drive_term: Callable[[float], float] = lambda t: drive_strength * enveloped_cos(
        t,
        omega_d=omega_d,
        t_ramp=t_ramp,
        envelope_func=envelope_func,
        envelope_func_args=envelope_func_args,
    )

    def dxdt(t: float, x: complex) -> complex:
        return -1j * omega_q * x + drive_term(t)

    sol = sp.integrate.solve_ivp(
        dxdt,
        (t_sample[0], t_sample[-1]),
        [init_displacement],
        t_eval=t_sample,
        rtol=1e-10,
        atol=1e-10,
    )
    return sol
