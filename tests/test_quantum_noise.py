from __future__ import annotations

import numpy as np

from ztpcraft.decoherence.quantum_noise import QuantumNoiseSpectralDensity


def test_quantum_noise_zero_temperature_absorption_suppressed() -> None:
    qn = QuantumNoiseSpectralDensity(alpha=1.2e-5, s=1.0, temperature=0.0)
    omega = 2.0 * np.pi * 5.0e9
    assert np.isclose(qn.S(-omega), 0.0)
    assert qn.S(omega) > 0.0


def test_quantum_noise_cutoff_behaviors() -> None:
    omega = 2.0 * np.pi * 4.0e9

    qn_hard = QuantumNoiseSpectralDensity(
        alpha=1.0,
        s=1.0,
        temperature=0.02,
        cutoff_type="hard",
        cutoff_freq=2.0 * np.pi * 3.0e9,
    )
    assert np.isclose(qn_hard.S(omega), 0.0)

    qn_exp = QuantumNoiseSpectralDensity(
        alpha=1.0,
        s=1.0,
        temperature=0.02,
        cutoff_type="exponential",
        cutoff_freq=2.0 * np.pi * 10.0e9,
    )
    assert qn_exp.S(omega) > 0.0


def test_quantum_noise_vectorized_matches_scalar() -> None:
    qn = QuantumNoiseSpectralDensity(alpha=3.0e-6, s=0.8, temperature=0.03)
    omega = np.array(
        [
            -2.0 * np.pi * 6.0e9,
            -2.0 * np.pi * 2.0e9,
            0.0,
            2.0 * np.pi * 2.0e9,
            2.0 * np.pi * 6.0e9,
        ],
        dtype=np.float64,
    )
    vectorized = qn.S_array(omega)
    scalar = np.array([qn.S(float(w)) for w in omega], dtype=np.float64)
    assert np.allclose(vectorized, scalar)


def test_quantum_noise_S_temperature_override() -> None:
    qn = QuantumNoiseSpectralDensity(alpha=1e-5, s=1.0, temperature=0.05)
    omega = 2.0 * np.pi * 5.0e9
    s_default = qn.S(omega)
    s_cold = qn.S(omega, temperature=0.01)
    assert s_cold < s_default
    assert np.isclose(qn.S(omega, temperature=0.05), s_default)
