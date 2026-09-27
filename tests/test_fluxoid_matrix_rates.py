"""Physical and bookkeeping checks for the preferred matrix API."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import h, hbar, k

from ztpcraft.decoherence.fgr import compute_rate_matrix
from ztpcraft.decoherence.quantum_noise import OhmicLikeNoise
from ztpcraft.projects.fluxonium.multiloop_fluxonium import (
    FluxoidModelParams,
    FluxoidSector,
    TwoLoopFluxoidSystem,
    SectorBasis,
    RateSetup,
    jump_matrix,
    prepare_rates,
    calculate_state_rates,
    thermal_populations,
    aggregate_rates,
    calculate_sector_rates,
    build_global_states,
    build_jump_matrix,
)


def test_basis_roundtrip_and_missing_labels():
    a, b = FluxoidSector(-2, 0), FluxoidSector(3, 1)
    basis = SectorBasis((a, b), 3)
    for index in range(basis.size):
        assert basis.index(*basis.label(index)) == index
    assert basis.block(b) == slice(3, 6)
    with pytest.raises(KeyError):
        basis.block(FluxoidSector(0, 0))
    with pytest.raises(IndexError):
        basis.index(a, -1)
    with pytest.raises(IndexError):
        basis.label(6)
    with pytest.raises(ValueError, match="Duplicate"):
        SectorBasis((a, a), 2)
    with pytest.raises(TypeError):
        SectorBasis((a,), 1.5)


def test_directed_jump_has_correct_rate_orientation():
    basis = SectorBasis((FluxoidSector(0, 0), FluxoidSector(1, 0)), 1)
    energies = np.array([[0.0], [2.0]])
    lowering = np.array([[0, 2j], [0, 0]])
    setup = RateSetup(basis, energies, lowering)
    rates = calculate_state_rates(setup, lambda omega, T: hbar**2, 0.1)
    np.testing.assert_allclose(rates, [[0, 0], [4, 0]])
    legacy_entry = compute_rate_matrix(
        energies.ravel(), lowering, lambda omega, T: hbar**2, 0.1
    )
    np.testing.assert_array_equal(legacy_entry, rates)


def test_thermal_balance_and_explicit_sector_average():
    basis = SectorBasis((FluxoidSector(0, 0), FluxoidSector(1, 0)), 2)
    energies = np.array([[0.0, 1.0], [0.3, 1.7]])
    coupling = np.array(
        [[0, 0, 1, 0.2j], [0, 0, 0.4, 0.7], [1, 0.4, 0, 0], [-0.2j, 0.7, 0, 0]]
    )
    setup = RateSetup(basis, energies, coupling)
    noise = OhmicLikeNoise(alpha=1e-80)
    T = 0.14
    rates = calculate_state_rates(setup, noise.S_array, T)
    probability = np.exp(-h * 1e9 * energies.ravel() / (k * T))
    probability /= probability.sum()
    flow = probability[:, None] * rates
    np.testing.assert_allclose(flow, flow.T, rtol=1e-12, atol=0)
    populations = thermal_populations(energies, T)
    sector = aggregate_rates(basis, rates, populations)
    expected = sum(
        populations[0, i] * rates[i, 2 + j] for i in range(2) for j in range(2)
    )
    assert sector[0, 1] == pytest.approx(expected)
    sector_probability = probability.reshape(2, 2).sum(axis=1)
    sector_flow = sector_probability[:, None] * sector
    np.testing.assert_allclose(sector_flow, sector_flow.T, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        thermal_populations(energies + np.array([[20], [-4]]), T), populations
    )
    np.testing.assert_allclose(calculate_sector_rates(setup, noise.S_array, T), sector)


def test_zero_frequency_ohmic_crossing_and_array_spectrum():
    noise = OhmicLikeNoise(alpha=1e-80, s=1)
    T = 0.14
    expected = noise.alpha * k * T / hbar
    assert noise.S(0.0, T) == pytest.approx(expected, rel=1e-12, abs=0)
    np.testing.assert_allclose(
        noise.S_array(np.array([[-1.0, 0.0, 1.0]]), T), expected, rtol=1e-9, atol=0
    )
    basis = SectorBasis((FluxoidSector(0, 0), FluxoidSector(1, 0)), 1)
    setup = RateSetup(basis, np.zeros((2, 1)), np.array([[0, 1], [1, 0]]))
    rates = calculate_state_rates(setup, noise.S_array, T)
    assert rates[0, 1] == pytest.approx(expected / hbar**2)
    assert rates[1, 0] == rates[0, 1]
    assert np.all(np.diag(rates) == 0)


def test_bad_spectrum_is_not_retried_or_hidden():
    basis = SectorBasis((FluxoidSector(0, 0),), 1)
    setup = RateSetup(basis, np.zeros((1, 1)), np.ones((1, 1)))
    calls = []

    def broken(omega, T):
        calls.append(omega.shape)
        raise RuntimeError("intentional noise failure")

    with pytest.raises(RuntimeError, match="intentional noise failure"):
        calculate_state_rates(setup, broken, 0.1)
    assert calls == [(1, 1)]
    with pytest.raises(ValueError, match="nonnegative"):
        calculate_state_rates(setup, lambda omega, T: -1.0, 0.1)


def test_setup_snapshot_and_aggregation_contract():
    basis = SectorBasis((FluxoidSector(0, 0),), 2)
    energies = np.zeros((1, 2))
    matrix = np.ones((2, 2))
    setup = RateSetup(basis, energies, matrix)
    energies[:] = 5
    matrix[:] = 0
    assert setup.energies[0, 0] == 0
    assert setup.coupling[0, 0] == 1
    with pytest.raises(ValueError):
        setup.energies[0, 0] = 5
    with pytest.raises(ValueError, match="sum to one"):
        aggregate_rates(basis, np.ones((2, 2)), np.ones((1, 2)))
    # Within-sector relaxation does not appear as a sector escape rate.
    np.testing.assert_array_equal(
        aggregate_rates(basis, np.ones((2, 2)), np.array([[0.2, 0.8]])), [[0.0]]
    )


def test_real_system_jump_and_constant_bias_reuse():
    params = FluxoidModelParams(0.46307, 0.43693, 5.1, 0.812, 0.0, 0.0, 1.0)
    basis = SectorBasis(tuple(FluxoidSector(m, 0) for m in [-1, 0, 1]), 4)
    system = TwoLoopFluxoidSystem(params, cutoff=25, evals_count=4)
    jump = jump_matrix(system, basis, (1, 0))
    old_states = build_global_states(basis.sectors, system)
    old_jump = build_jump_matrix(system, old_states, (1, 0))
    np.testing.assert_allclose(jump, old_jump, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        jump_matrix(system, basis, (-1, 0)), jump.conj().T, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_array_equal(jump_matrix(system, basis, (0, 1)), 0)
    setup = prepare_rates(system, basis, jump + jump.conj().T)
    phi_c = 2 * np.pi * 0.37
    total, fa, fb = system.model.inductive_fractions()
    phi_d = (fb - fa) * phi_c
    other = TwoLoopFluxoidSystem(
        replace(params, phi_ext_a=(phi_c + phi_d) / 2, phi_ext_b=(phi_c - phi_d) / 2),
        cutoff=25,
        evals_count=4,
    )
    other_jump = jump_matrix(other, basis, (1, 0))
    direct = prepare_rates(other, basis, other_jump + other_jump.conj().T)
    offsets = np.array(
        [
            other.model.energy_offset(s) - system.model.energy_offset(s)
            for s in basis.sectors
        ]
    )
    reused = RateSetup(basis, setup.energies + offsets[:, None], setup.coupling)
    np.testing.assert_allclose(reused.energies, direct.energies, rtol=1e-12, atol=1e-12)
    noise = OhmicLikeNoise(alpha=1e-80)
    np.testing.assert_allclose(
        calculate_sector_rates(reused, noise.S_array, 0.14),
        calculate_sector_rates(direct, noise.S_array, 0.14),
        rtol=1e-10,
        atol=1e-15,
    )
