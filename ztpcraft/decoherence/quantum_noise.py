"""Quantum noise spectral density models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar, k

FloatArray = NDArray[np.float64]
CutoffType = Literal["exponential", "hard", None]


@dataclass
class OhmicLikeNoise:
    """
    Unsymmetrized quantum, ohmic-like noise spectral density S(omega).

    J(omega) = alpha * omega^s if 0 < omega < cutoff_freq,
             = 0 if omega <= 0,
             = alpha * omega^s * cutoff_function(omega) if omega > cutoff_freq.
    where cutoff_function(omega) is either
        - exponential: exp(-omega / cutoff_freq)
        - hard: 1 if omega < cutoff_freq, 0 otherwise
        - None: 1 for all

    S(omega) = J(omega) * (bose(omega) + 1) if omega > 0,
             = J(omega) * bose(omega) if omega < 0.

    The cutoff_function is used to smoothly transition the spectral density to 0 at the cutoff frequency.

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
        elif (
            self.cutoff_type == "hard"
            and self.cutoff_freq is not None
            and omega > self.cutoff_freq
        ):
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


@dataclass
class CapacitiveNoise:
    """
    Unsymmetrized quantum, capacitive noise spectral density S(omega).

    S(omega) = 2*hbar/C/Q_cap * coth(hbar*|omega|/2kT)/(1+exp(-hbar*omega/kT))

    Q_cap can be a callable function of omega, or can be a fixed value.
    """

    C: float
    Q_cap: float | Callable[[float], float]
    temperature: float = 0.0

    def _thermal_temperature(self, temperature: float | None) -> float:
        return float(self.temperature if temperature is None else temperature)

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
        beta = hbar * omega / (k * self._thermal_temperature(temperature))
        if abs(beta) > 100:
            coth_factor = 1.0
            if beta > 0:
                one_over_exp_plus_one = 1.0
            else:
                one_over_exp_plus_one = 0.0
        else:
            coth_factor = 1 / np.tanh(np.abs(beta) / 2)
            one_over_exp_plus_one = 1 / (np.exp(-beta) + 1)
        Q_cap_value = self.Q_cap(omega) if callable(self.Q_cap) else self.Q_cap
        noise_spectral_density = (
            2 * hbar / self.C / Q_cap_value * coth_factor * one_over_exp_plus_one
        )
        return noise_spectral_density

    def S_array(
        self,
        omega: NDArray[np.float64] | list[float],
        temperature: float | None = None,
    ) -> FloatArray:
        """Vectorized unsymmetrized spectral density over SI angular frequencies."""
        omega_array = np.asarray(omega, dtype=np.float64)
        T = self._thermal_temperature(temperature)

        coth_factor = 1.0 / np.tanh(hbar * np.abs(omega_array) / (2.0 * k * T))
        exp_plus_one = np.exp(-hbar * omega_array / (k * T)) + 1.0

        if callable(self.Q_cap):
            q_cap_values = np.asarray(
                [self.Q_cap(float(w)) for w in omega_array], dtype=np.float64
            )
        else:
            q_cap_values = float(self.Q_cap)

        result = 2.0 * hbar / self.C / q_cap_values * coth_factor / exp_plus_one
        return np.asarray(result, dtype=np.float64)

@dataclass
class InductiveNoise:
    """
    Unsymmetrized quantum, inductive noise spectral density S(omega).

    S(omega) = 2*hbar/L/Q_ind * coth(hbar*|omega|/2kT)/(1+exp(-hbar*omega/kT))

    Q_ind can be a callable function of omega, or can be a fixed value.
    """

    L: float
    Q_ind: float | Callable[[float], float]
    temperature: float = 0.0

    def _thermal_temperature(self, temperature: float | None) -> float:
        return float(self.temperature if temperature is None else temperature)

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
        beta = hbar * omega / (k * self._thermal_temperature(temperature))
        if abs(beta) > 100:
            coth_factor = 1.0
            if beta > 0:
                one_over_exp_plus_one = 1.0
            else:
                one_over_exp_plus_one = 0.0
        else:
            coth_factor = 1 / np.tanh(np.abs(beta) / 2)
            one_over_exp_plus_one = 1 / (np.exp(-beta) + 1)
        Q_ind_value = self.Q_ind(omega) if callable(self.Q_ind) else self.Q_ind
        noise_spectral_density = (
            2 * hbar / self.L / Q_ind_value * coth_factor * one_over_exp_plus_one
        )
        return noise_spectral_density

    def S_array(
        self,
        omega: NDArray[np.float64] | list[float],
        temperature: float | None = None,
    ) -> FloatArray:
        """Vectorized unsymmetrized spectral density over SI angular frequencies."""
        omega_array = np.asarray(omega, dtype=np.float64)
        T = self._thermal_temperature(temperature)

        coth_factor = 1.0 / np.tanh(hbar * np.abs(omega_array) / (2.0 * k * T))
        exp_plus_one = np.exp(-hbar * omega_array / (k * T)) + 1.0

        if callable(self.Q_cap):
            q_cap_values = np.asarray(
                [self.Q_cap(float(w)) for w in omega_array], dtype=np.float64
            )
        else:
            q_cap_values = float(self.Q_cap)

        result = 2.0 * hbar / self.C / q_cap_values * coth_factor / exp_plus_one
        return np.asarray(result, dtype=np.float64)