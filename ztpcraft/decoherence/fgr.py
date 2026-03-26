"""Fermi-Golden-Rule helpers for decoherence calculations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.constants import hbar, k

__all__ = ["calc_therm_ratio", "fgr_decay_rate"]

FrequencyUnit = Literal["GHz", "MHz", "kHz", "Hz"]
_FREQUENCY_SCALE_HZ: dict[FrequencyUnit, float] = {
    "GHz": 1e9,
    "MHz": 1e6,
    "kHz": 1e3,
    "Hz": 1.0,
}


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
) -> float:
    """Compute a transition rate from Fermi's Golden Rule.

    Energies are assumed to be in linear-frequency units and internally
    converted to angular frequency via ``omega = 2*pi*(Ei - Ej)``.
    """
    omega = 2.0 * np.pi * (energy_i - energy_j)

    spectral_weight = _evaluate_spectral_density(spectral_density, omega, T)
    if include_upward:
        spectral_weight += _evaluate_spectral_density(spectral_density, -omega, T)

    rate = (abs(matrix_element) ** 2) * spectral_weight
    real_rate = np.real_if_close(rate)
    if np.iscomplexobj(real_rate):
        raise ValueError("Computed FGR rate is complex; spectral density must be real.")
    return float(real_rate)
