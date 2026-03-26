import numpy as np

from scipy.constants import hbar, k

from ztpcraft.decoherence.fgr import calc_therm_ratio, fgr_decay_rate


def test_fgr_decay_rate_downward_only() -> None:
    def spectral_density(omega: float, temperature: float | None = None) -> float:
        _ = temperature
        return 0.1 + 0.05 * abs(omega)

    rate = fgr_decay_rate(
        energy_i=6.0,
        energy_j=5.0,
        matrix_element=2.0,
        spectral_density=spectral_density,
        include_upward=False,
    )
    expected = (abs(2.0) ** 2) * (0.1 + 0.05 * (2.0 * np.pi))
    assert np.isclose(rate, expected)


def test_fgr_decay_rate_include_upward_with_temperature() -> None:
    def spectral_density(omega: float, temperature: float | None = None) -> float:
        temp = 0.0 if temperature is None else temperature
        return abs(omega) + 2.0 * temp

    rate = fgr_decay_rate(
        energy_i=4.0,
        energy_j=1.0,
        matrix_element=1.5,
        spectral_density=spectral_density,
        T=0.2,
        include_upward=True,
    )
    omega = 2.0 * np.pi * (4.0 - 1.0)
    expected = (abs(1.5) ** 2) * ((abs(omega) + 0.4) + (abs(-omega) + 0.4))
    assert np.isclose(rate, expected)


def test_fgr_accepts_one_argument_spectral_density() -> None:
    def spectral_density(omega: float) -> float:
        return 0.2 * abs(omega)

    rate = fgr_decay_rate(
        energy_i=2.5,
        energy_j=2.0,
        matrix_element=3.0,
        spectral_density=spectral_density,
        T=0.1,
        include_upward=False,
    )
    expected = (abs(3.0) ** 2) * (0.2 * abs(2.0 * np.pi * 0.5))
    assert np.isclose(rate, expected)


def test_calc_therm_ratio_unit_scaling() -> None:
    omega_ghz = 2.0 * np.pi * 5.0  # angular frequency in GHz units
    ratio = calc_therm_ratio(omega_ghz, T=0.05, units="GHz")
    expected = (hbar * omega_ghz * 1e9) / (k * 0.05)
    assert np.isclose(ratio, expected)


def test_calc_therm_ratio_requires_positive_temperature() -> None:
    try:
        calc_therm_ratio(omega=1.0, T=0.0)
    except ValueError:
        return
    raise AssertionError("calc_therm_ratio should reject non-positive temperature")
