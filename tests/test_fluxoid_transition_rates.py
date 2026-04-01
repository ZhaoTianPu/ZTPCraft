from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_transition_rates import (
    FluxoidOperator,
    GlobalState,
    SectorJumpOperator,
    build_energy_array,
    build_global_states,
    build_jump_matrix,
    build_operator_matrix,
    compute_rate_matrix,
    compute_all_decay_rates,
    sector_rate_matrix,
    sector_rate_matrix_fast,
    sector_to_sector_rate,
    thermal_weights,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass
class FakeFluxoidSystem:
    energies: dict[FluxoidSector, FloatArray]
    overlaps: dict[tuple[FluxoidSector, FluxoidSector], ComplexArray]

    def eigenvalues_with_offset(self, sector: FluxoidSector) -> FloatArray:
        return self.energies[sector]

    def overlap_matrix(self, sector1: FluxoidSector, sector2: FluxoidSector) -> ComplexArray:
        if (sector1, sector2) in self.overlaps:
            return self.overlaps[(sector1, sector2)]
        n1 = len(self.energies[sector1])
        n2 = len(self.energies[sector2])
        return np.zeros((n1, n2), dtype=np.complex128)


def _build_fixture() -> tuple[
    FakeFluxoidSystem, list[FluxoidSector], list[GlobalState], FluxoidOperator, FluxoidOperator
]:
    s00 = FluxoidSector(0, 0)
    s10 = FluxoidSector(1, 0)
    s01 = FluxoidSector(0, 1)
    s11 = FluxoidSector(1, 1)
    sectors = [s00, s10, s01, s11]

    energies = {
        s00: np.array([0.00, 0.90], dtype=np.float64),
        s10: np.array([0.35, 1.25], dtype=np.float64),
        s01: np.array([0.45, 1.40], dtype=np.float64),
        s11: np.array([0.75, 1.80], dtype=np.float64),
    }

    ab = np.array([[0.70, 0.12], [0.08, 0.62]], dtype=np.complex128)
    ba = np.array([[0.66, 0.11], [0.06, 0.57]], dtype=np.complex128)
    cd = np.array([[0.52, 0.09], [0.07, 0.45]], dtype=np.complex128)
    dc = np.array([[0.49, 0.08], [0.05, 0.41]], dtype=np.complex128)
    ac = np.array([[0.61, 0.10], [0.04, 0.50]], dtype=np.complex128)
    ca = np.array([[0.58, 0.13], [0.06, 0.46]], dtype=np.complex128)
    bd = np.array([[0.55, 0.14], [0.05, 0.48]], dtype=np.complex128)
    db = np.array([[0.53, 0.12], [0.07, 0.43]], dtype=np.complex128)

    overlaps = {
        (s10, s00): ab,
        (s00, s10): ba,
        (s11, s01): cd,
        (s01, s11): dc,
        (s01, s00): ac,
        (s00, s01): ca,
        (s11, s10): bd,
        (s10, s11): db,
    }

    system = FakeFluxoidSystem(energies=energies, overlaps=overlaps)
    states = build_global_states(sectors, system)  # type: ignore[arg-type]

    raise_a = SectorJumpOperator(system=system, delta=(1, 0))  # type: ignore[arg-type]
    lower_a = SectorJumpOperator(system=system, delta=(-1, 0))  # type: ignore[arg-type]
    return system, sectors, states, raise_a, lower_a


def _state(sector: FluxoidSector, level: int) -> GlobalState:
    return GlobalState(sector=sector, level=level)


def _legacy_rates(
    system: FakeFluxoidSystem,
    states: list[GlobalState],
    operator: FluxoidOperator,
    spectral_density,
    temperature: float,
) -> dict[tuple[int, int], float]:
    from ztpcraft.decoherence.fgr import fgr_decay_rate

    bound = operator.bind_states(states)
    energies = build_energy_array(system, states)  # type: ignore[arg-type]
    rates: dict[tuple[int, int], float] = {}
    for i, s_i in enumerate(states):
        for j, s_j in enumerate(states):
            if i == j:
                continue
            element = bound.matrix_element(s_i, s_j)
            if abs(element) < 1e-14:
                continue
            rates[(i, j)] = fgr_decay_rate(
                energy_i=float(energies[i]),
                energy_j=float(energies[j]),
                matrix_element=element,
                spectral_density=spectral_density,
                T=temperature,
            )
    return rates


def test_single_sum_product_composite_and_hermitian_operators() -> None:
    system, sectors, states, raise_a, lower_a = _build_fixture()
    s00, s10, s01, _ = sectors
    lower_b = SectorJumpOperator(system=system, delta=(0, -1))  # type: ignore[arg-type]
    raise_b = SectorJumpOperator(system=system, delta=(0, 1))  # type: ignore[arg-type]

    single = raise_a.matrix_element(_state(s10, 0), _state(s00, 1))
    expected_single = system.overlap_matrix(s10, s00)[0, 1]
    assert np.isclose(single, expected_single)

    summed = (raise_a + lower_a).matrix_element(_state(s10, 0), _state(s00, 1))
    expected_sum = raise_a.matrix_element(_state(s10, 0), _state(s00, 1)) + lower_a.matrix_element(
        _state(s10, 0), _state(s00, 1)
    )
    assert np.isclose(summed, expected_sum)

    product = (raise_a * lower_b).bind_states(states)
    bra = _state(s10, 1)
    ket = _state(s01, 0)
    product_me = product.matrix_element(bra, ket)
    manual_product = 0.0j
    for k_state in states:
        manual_product += raise_a.matrix_element(bra, k_state) * lower_b.matrix_element(
            k_state, ket
        )
    assert np.isclose(product_me, manual_product)

    composite = ((raise_a + lower_a) * (raise_b + lower_b)).bind_states(states)
    composite_me = composite.matrix_element(_state(s10, 0), _state(s00, 1))
    manual_composite = 0.0j
    for k_state in states:
        manual_composite += (raise_a + lower_a).matrix_element(
            _state(s10, 0), k_state
        ) * (raise_b + lower_b).matrix_element(k_state, _state(s00, 1))
    assert np.isclose(composite_me, manual_composite)

    op = raise_a + (0.35 - 0.2j) * lower_b
    hermitian = op + op.dagger()
    hij = hermitian.matrix_element(_state(s10, 0), _state(s00, 1))
    expected_hij = op.matrix_element(_state(s10, 0), _state(s00, 1)) + np.conj(
        op.matrix_element(_state(s00, 1), _state(s10, 0))
    )
    assert np.isclose(hij, expected_hij)


def test_vectorized_operator_build_and_rate_matrix_matches_legacy() -> None:
    system, sectors, states, raise_a, lower_a = _build_fixture()
    _, _, s01, _ = sectors
    lower_b = SectorJumpOperator(system=system, delta=(0, -1))  # type: ignore[arg-type]
    raise_b = SectorJumpOperator(system=system, delta=(0, 1))  # type: ignore[arg-type]
    operator = (raise_a + lower_a) * (raise_b + lower_b)

    def spectral_density(omega: float, temperature: float | None = None) -> float:
        t_shift = 0.0 if temperature is None else 0.1 * temperature
        return abs(omega) + 0.2 + t_shift

    energies = build_energy_array(system, states)  # type: ignore[arg-type]
    jump = build_jump_matrix(system, states, delta=(1, 0))  # type: ignore[arg-type]
    assert jump.shape == (len(states), len(states))

    op_matrix = build_operator_matrix(operator, system, states)  # type: ignore[arg-type]
    rates_matrix = compute_rate_matrix(
        energies=energies,
        O_matrix=op_matrix,
        spectral_density=spectral_density,
        T=0.06,
    )
    assert rates_matrix.shape == (len(states), len(states))
    assert np.allclose(np.diag(rates_matrix), 0.0)

    fast_rates = compute_all_decay_rates(
        system=system,  # type: ignore[arg-type]
        states=states,
        operator=operator,
        spectral_density=spectral_density,
        T=0.06,
    )
    slow_rates = _legacy_rates(
        system=system,
        states=states,
        operator=operator,
        spectral_density=spectral_density,
        temperature=0.06,
    )
    assert set(fast_rates.keys()) == set(slow_rates.keys())
    for key in fast_rates:
        assert np.isclose(fast_rates[key], slow_rates[key], rtol=1e-12, atol=1e-12)

    # Sanity-check one off-diagonal transition exists for this composite operator.
    transitions_from_s01 = [
        key for key in fast_rates if states[key[1]].sector == s01 and states[key[0]].sector != s01
    ]
    assert len(transitions_from_s01) > 0


def test_thermal_weights_and_sector_aggregation_and_detailed_balance() -> None:
    system, sectors, states, raise_a, lower_a = _build_fixture()
    s00, s10, _, _ = sectors
    operator = raise_a + lower_a

    def spectral_density(omega: float, temperature: float | None = None) -> float:
        t_shift = 0.0 if temperature is None else 0.1 * temperature
        return abs(omega) + 0.2 + t_shift

    weights = thermal_weights(system, s00, T=0.06)  # type: ignore[arg-type]
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1]

    direct = sector_to_sector_rate(
        system=system,  # type: ignore[arg-type]
        sector_from=s00,
        sector_to=s10,
        states=states,
        operator=operator,
        spectral_density=spectral_density,
        T=0.06,
    )

    energies = build_energy_array(system, states)  # type: ignore[arg-type]
    op_matrix = build_operator_matrix(operator, system, states)  # type: ignore[arg-type]
    rates = compute_rate_matrix(
        energies=energies,
        O_matrix=op_matrix,
        spectral_density=spectral_density,
        T=0.06,
    )
    thermal_by_sector = {
        sector: thermal_weights(system, sector, T=0.06)  # type: ignore[arg-type]
        for sector in sectors
    }
    matrix_fast = sector_rate_matrix_fast(
        states=states, sectors=sectors, rates=rates, weights=thermal_by_sector
    )
    matrix = sector_rate_matrix(
        system=system,  # type: ignore[arg-type]
        sectors=sectors,
        states=states,
        operator=operator,
        spectral_density=spectral_density,
        T=0.06,
    )
    assert np.isclose(matrix[0, 1], direct)
    assert np.allclose(matrix, matrix_fast, rtol=1e-12, atol=1e-12)
    assert matrix.shape == (len(sectors), len(sectors))
    assert np.all(np.isreal(matrix))

