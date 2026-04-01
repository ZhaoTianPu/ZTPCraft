"""Quantum noise spectral density models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar, k

FloatArray = NDArray[np.float64]
CutoffType = Literal["exponential", "hard", None]


@dataclass
class QuantumNoiseSpectralDensity:
    """
    Unsymmetrized quantum noise spectral density S(omega).

    All frequencies are angular frequencies in SI units (rad/s).
    """

    alpha: float
    s: float = 1.0
    temperature: float = 0.0
    cutoff_type: CutoffType = "exponential"
    cutoff_freq: float | None = None

    def _thermal_temperature(self, temperature: float | None) -> float:
        return float(self.temperature if temperature is None else temperature)

    def bose(self, omega: float, temperature: float | None = None) -> float:
        """Bose occupation n(omega) = 1 / (exp(hbar*omega/kT) - 1)."""
        T = self._thermal_temperature(temperature)
        if T <= 0.0:
            return 0.0

        beta = hbar * abs(float(omega)) / (k * T)
        if beta > 700.0:
            return 0.0
        return float(1.0 / (np.exp(beta) - 1.0))

    def J(self, omega: float) -> float:
        """Spectral function J(omega) for omega > 0."""
        omega = float(omega)
        if omega <= 0.0:
            raise ValueError("J(omega) expects omega > 0.")

        value = self.alpha * omega**self.s
        if self.cutoff_type == "exponential" and self.cutoff_freq is not None:
            value *= float(np.exp(-omega / self.cutoff_freq))
        elif self.cutoff_type == "hard" and self.cutoff_freq is not None and omega > self.cutoff_freq:
            return 0.0
        return float(value)

    def S(self, omega: float, temperature: float | None = None) -> float:
        """
        Unsymmetrized spectral density.

        omega > 0: emission branch
        omega < 0: absorption branch

        If ``temperature`` is None, uses ``self.temperature``. A non-None value
        overrides for this call only (compatible with FGR passing ``T`` as the
        second positional argument).
        """
        omega = float(omega)
        if omega == 0.0:
            return 0.0

        if omega > 0.0:
            return float(self.J(omega) * (self.bose(omega, temperature) + 1.0))
        omega_abs = abs(omega)
        return float(self.J(omega_abs) * self.bose(omega_abs, temperature))

    def S_array(
        self,
        omega: NDArray[np.float64] | list[float],
        temperature: float | None = None,
    ) -> FloatArray:
        """Vectorized unsymmetrized spectral density over SI angular frequencies."""
        omega_array = np.asarray(omega, dtype=np.float64)
        result = np.zeros_like(omega_array, dtype=np.float64)

        pos = omega_array > 0.0
        neg = omega_array < 0.0
        if np.any(pos):
            values_pos = np.asarray(
                [
                    self.J(float(w)) * (self.bose(float(w), temperature) + 1.0)
                    for w in omega_array[pos]
                ],
                dtype=np.float64,
            )
            result[pos] = values_pos
        if np.any(neg):
            values_neg = np.asarray(
                [
                    self.J(float(abs(w))) * self.bose(float(abs(w)), temperature)
                    for w in omega_array[neg]
                ],
                dtype=np.float64,
            )
            result[neg] = values_neg
        return result

