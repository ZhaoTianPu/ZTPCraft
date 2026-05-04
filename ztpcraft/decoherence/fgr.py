"""Fermi-Golden-Rule helpers for decoherence calculations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar, k

__all__ = ["calc_therm_ratio", "fgr_decay_rate", "compute_rate_matrix"]

FrequencyUnit = Literal["GHz", "MHz", "kHz", "Hz"]
_FREQUENCY_SCALE_HZ: dict[FrequencyUnit, float] = {
    "GHz": 1e9,
    "MHz": 1e6,
    "kHz": 1e3,
    "Hz": 1.0,
}
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _spectral_density_grid(
    spectral_density: Callable[..., float | complex],
    omega: FloatArray,
    temperature: float | None,
) -> ComplexArray:
    """Evaluate spectral density on an omega grid with optional temperature."""
    supports_temp = _supports_temperature_argument(spectral_density)

    if temperature is None or not supports_temp:
        try:
            values = spectral_density(omega)
            return np.asarray(values, dtype=np.complex128)
        except Exception:
            vectorized = np.vectorize(spectral_density, otypes=[np.complex128])
            return np.asarray(vectorized(omega), dtype=np.complex128)

    try:
        values = spectral_density(omega, temperature)
        return np.asarray(values, dtype=np.complex128)
    except Exception:

        def _scalar_eval(w: float) -> complex:
            return complex(spectral_density(w, temperature))

        vectorized = np.vectorize(_scalar_eval, otypes=[np.complex128])
        return np.asarray(vectorized(omega), dtype=np.complex128)


def _supports_temperature_argument(
    spectral_density: Callable[..., float | complex],
) -> bool:
    """Return True if ``spectral_density`` can accept a second positional arg."""
    try:
        signature = inspect.signature(spectral_density)
    except (TypeError, ValueError):
        # Some compiled/builtin callables do not expose signatures.
        return True

    params = tuple(signature.parameters.values())
    if any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in params):
        return True

    positional = [
        param
        for param in params
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _evaluate_spectral_density(
    spectral_density: Callable[..., float | complex],
    omega: float,
    temperature: float | None,
) -> float | complex:
    """Evaluate spectral density with flexible callable signatures."""
    if temperature is None or not _supports_temperature_argument(spectral_density):
        return spectral_density(omega)
    return spectral_density(omega, temperature)


def calc_therm_ratio(omega: float, T: float, units: FrequencyUnit = "GHz") -> float:
    r"""Compute the thermal ratio :math:`\beta\hbar\omega`.

    Parameters
    ----------
    omega:
        Angular frequency expressed in ``units``.
    T:
        Temperature in Kelvin. Must be positive.
    units:
        Frequency unit scale associated with ``omega``.

    Returns
    -------
    float
        The dimensionless quantity :math:`\hbar\omega / (k_B T)`.
    """
    if T <= 0.0:
        raise ValueError("Temperature T must be positive.")

    omega_si = omega * _FREQUENCY_SCALE_HZ[units]
    return float((hbar * omega_si) / (k * T))


def fgr_decay_rate(
    energy_i: float,
    energy_j: float,
    matrix_element: float | complex,
    spectral_density: Callable[..., float | complex],
    T: float | None = None,
    include_upward: bool = False,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
) -> float:
    """Compute a transition rate from Fermi's Golden Rule.

    Energies are assumed to be in linear-frequency units (``units``) and internally
    converted to angular frequency via ``omega = 2*pi*(Ei - Ej)``.
    By default, the spectral density is evaluated in SI angular-frequency units
    (rad/s), i.e. ``omega * scale_hz[units]``.
    """
    omega = 2.0 * np.pi * (energy_i - energy_j)
    omega_for_spectral = omega
    if spectral_omega_units == "SI":
        omega_for_spectral = omega * _FREQUENCY_SCALE_HZ[units]

    spectral_weight = _evaluate_spectral_density(
        spectral_density, omega_for_spectral, T
    )
    if include_upward:
        spectral_weight += _evaluate_spectral_density(
            spectral_density, -omega_for_spectral, T
        )

    rate = (abs(matrix_element) ** 2) * spectral_weight / hbar**2
    real_rate = np.real_if_close(rate)
    if np.iscomplexobj(real_rate):
        raise ValueError("Computed FGR rate is complex; spectral density must be real.")
    return float(real_rate)


def compute_rate_matrix(
    energies: FloatArray,
    O_matrix: ComplexArray,
    spectral_density: Callable[..., float | complex],
    T: float | None,
    units: FrequencyUnit = "GHz",
    spectral_omega_units: Literal["input", "SI"] = "SI",
    matrix_element_cutoff: float = 1e-14,
) -> FloatArray:
    """Compute dense FGR transition-rate matrix from energies and operator matrix."""
    ei = energies[:, None]
    ej = energies[None, :]
    omega = 2.0 * np.pi * (ei - ej)
    omega_for_spectral = omega
    if spectral_omega_units == "SI":
        omega_for_spectral = omega * _FREQUENCY_SCALE_HZ[units]
    spectral = _spectral_density_grid(spectral_density, omega_for_spectral, T)

    matrix_element_mod_square = np.abs(O_matrix) ** 2
    matrix_element_mod_square[
        matrix_element_mod_square < matrix_element_cutoff**2
    ] = 0.0
    rates_complex = (matrix_element_mod_square * spectral) / (hbar**2)
    np.fill_diagonal(rates_complex, 0.0)
    rates_real = np.real_if_close(rates_complex)
    if np.iscomplexobj(rates_real):
        raise ValueError("Computed FGR rates are complex.")

    rates = np.asarray(rates_real, dtype=np.float64)
    return rates
