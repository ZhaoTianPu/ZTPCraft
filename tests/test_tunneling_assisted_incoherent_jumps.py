from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

from ztpcraft.decoherence.fgr import fgr_decay_rate
from ztpcraft.projects.fluxonium.multiloop_fluxonium.tunneling_assisted_incoherent_jumps import (
    NoiseChannel,
    PhaseSlipSector,
    SpectrumInAPhaseSlipSector,
    PhaseSlipStateLabel,
    TunnelingAssistedIncoherentJumps,
    TunnelingAssistedModelParams,
    ensure_explicit_p_values,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class _FakeTunnelingModel(TunnelingAssistedIncoherentJumps):
    def __init__(
        self,
        params: TunnelingAssistedModelParams,
        p_values: list[int],
        energies_by_p: dict[int, FloatArray],
        overlap_by_pair: dict[tuple[int, int], ComplexArray],
        operator_by_p: dict[int, ComplexArray],
        threshold: float = 0.2,
    ):
        super().__init__(
            params=params,
            p_values=p_values,
            cutoff=max(len(e) for e in energies_by_p.values()),
            evals_count=max(len(e) for e in energies_by_p.values()),
            perturbation_ratio_threshold=threshold,
        )
        self._energies_by_p = energies_by_p
        self._overlap_by_pair = overlap_by_pair
        self._operator_by_p = operator_by_p

    def energy_offset(self, sector: PhaseSlipSector) -> float:  # type: ignore[override]
        return 0.0

    def get_bare_spectrum_in_a_sector(
        self, sector: PhaseSlipSector
    ) -> SpectrumInAPhaseSlipSector:  # type: ignore[override]
        evals = self._energies_by_p[sector.P]
        dim = len(evals)
        return SpectrumInAPhaseSlipSector(
            sector=sector,
            evals=np.asarray(evals, dtype=np.float64),
            evecs=np.eye(dim, dtype=np.complex128),
            shift=0.0,
            cutoff=dim,
            osc=None,  # type: ignore[arg-type]
            fluxonium=None,
        )

    def get_state_overlap_matrix(
        self, sector1: PhaseSlipSector, sector2: PhaseSlipSector
    ) -> ComplexArray:  # type: ignore[override]
        key = (sector1.P, sector2.P)
        if key in self._overlap_by_pair:
            return self._overlap_by_pair[key]
        n1 = len(self._energies_by_p[sector1.P])
        n2 = len(self._energies_by_p[sector2.P])
        return np.zeros((n1, n2), dtype=np.complex128)

    def _single_sector_operator_in_bare_eigenbasis(  # type: ignore[override]
        self, sector: PhaseSlipSector, operator: Any
    ):
        return self._operator_by_p[sector.P]


def _build_fixture(
    E_S: complex,
    threshold: float = 0.2,
    energies_override: dict[int, FloatArray] | None = None,
) -> _FakeTunnelingModel:
    p_values = [-1, 0, 1]
    energies_by_p = {
        -1: np.array([0.20, 1.10], dtype=np.float64),
        0: np.array([0.00, 0.90], dtype=np.float64),
        1: np.array([0.35, 1.25], dtype=np.float64),
    }
    if energies_override is not None:
        energies_by_p.update(energies_override)
    overlap_by_pair = {
        (1, 0): np.array([[0.7, 0.1], [0.05, 0.6]], dtype=np.complex128),
        (0, 1): np.array([[0.66, 0.08], [0.03, 0.58]], dtype=np.complex128),
        (-1, 0): np.array([[0.6, 0.12], [0.07, 0.5]], dtype=np.complex128),
        (0, -1): np.array([[0.57, 0.11], [0.06, 0.47]], dtype=np.complex128),
    }
    operator_by_p = {
        -1: np.array([[0.2, 0.1], [0.1, -0.1]], dtype=np.complex128),
        0: np.array([[0.3, -0.08], [-0.08, 0.05]], dtype=np.complex128),
        1: np.array([[0.22, 0.09], [0.09, -0.07]], dtype=np.complex128),
    }
    params = TunnelingAssistedModelParams(
        EL_a=0.4,
        EL_b=0.5,
        EJ=7.5,
        EC=1.3,
        phi_ext_a=0.1,
        phi_ext_b=0.2,
        E_S=E_S,
    )
    return _FakeTunnelingModel(
        params=params,
        p_values=p_values,
        energies_by_p=energies_by_p,
        overlap_by_pair=overlap_by_pair,
        operator_by_p=operator_by_p,
        threshold=threshold,
    )


def _channel_n(name: str, spectral_density) -> NoiseChannel:
    return NoiseChannel(
        name=name,
        operator="n_operator",
        spectral_density=spectral_density,
    )


def test_explicit_p_values_and_state_truncation() -> None:
    assert ensure_explicit_p_values([1, 0, 1, -1]) == [-1, 0, 1]

    model = _build_fixture(E_S=1e-3)
    states = model.build_states()
    assert {state.sector.P for state in states} == {-1, 0, 1}
    assert all(isinstance(state, PhaseSlipStateLabel) for state in states)


def test_zero_tunneling_has_no_cross_p_rates() -> None:
    model = _build_fixture(E_S=0.0)
    hybridization = model.build_perturbative_hybridization_info()

    def spectral_density(omega: float, temperature: float | None = None) -> float:
        return abs(omega) + (0.0 if temperature is None else 0.1 * temperature) + 0.5

    rates = model.compute_transition_rate_matrix_for_channel(
        hybridization=hybridization,
        channel=_channel_n("single", spectral_density),
        T=0.05,
    )
    states = hybridization.states
    for i, s_i in enumerate(states):
        for j, s_j in enumerate(states):
            if i == j:
                continue
            if s_i.sector.P != s_j.sector.P:
                assert np.isclose(rates[i, j], 0.0, atol=1e-16)


def test_rates_follow_es_squared_scaling() -> None:
    model_small = _build_fixture(E_S=2e-4)
    model_big = _build_fixture(E_S=4e-4)
    hybrid_small = model_small.build_perturbative_hybridization_info()
    hybrid_big = model_big.build_perturbative_hybridization_info()

    def spectral_density(omega: float, temperature: float | None = None) -> float:
        _ = temperature
        return abs(omega) + 0.4

    rate_small = model_small.p_to_p_rate(
        hybridization=hybrid_small,
        channel=_channel_n("single", spectral_density),
        p_from=0,
        p_to=1,
        T=0.05,
    )
    rate_big = model_big.p_to_p_rate(
        hybridization=hybrid_big,
        channel=_channel_n("single", spectral_density),
        p_from=0,
        p_to=1,
        T=0.05,
    )
    assert rate_small > 0.0
    assert np.isclose(rate_big / rate_small, 4.0, rtol=5e-2)


def test_hybridization_diagnostics_flags_breakdown() -> None:
    model = _build_fixture(
        E_S=1e-2,
        threshold=1e-2,
        energies_override={1: np.array([1e-7, 1.25], dtype=np.float64)},
    )

    hybridization = model.build_perturbative_hybridization_info()
    assert len(hybridization.invalid_contributions) > 0
    assert any(
        c.reason in {"hybridization_ratio_exceeds_threshold"}
        or (c.reason is not None and "detuning_floor" in c.reason)
        for c in hybridization.invalid_contributions
    )


def test_dressed_eigenvectors_in_bare_basis_contains_identity_plus_neighbor_components() -> (
    None
):
    model = _build_fixture(E_S=1e-3)
    hybridization = model.build_perturbative_hybridization_info()
    identity = np.eye(len(hybridization.states), dtype=np.complex128)
    delta = hybridization.dressed_eigenvectors_in_bare_basis - identity

    # Columns correspond to dressed state nearest each source bare state.
    # Self-component should remain exactly identity in this perturbative map.
    assert np.allclose(
        np.diag(hybridization.dressed_eigenvectors_in_bare_basis), 1.0 + 0.0j
    )

    for row, source in enumerate(hybridization.states):
        for col, partner in enumerate(hybridization.states):
            if abs(source.sector.P - partner.sector.P) > 1:
                assert np.isclose(delta[row, col], 0.0 + 0.0j)


def test_user_supplied_truncated_basis_limits_hybridization_targets() -> None:
    model = _build_fixture(E_S=1e-3)
    states = [
        PhaseSlipStateLabel(sector=PhaseSlipSector(P=-1), level=0),
        PhaseSlipStateLabel(sector=PhaseSlipSector(P=0), level=0),
    ]
    hybridization = model.build_perturbative_hybridization_info(states=states)
    assert len(hybridization.states) == 2

    # No P=+1 state in basis, so hybridization has nowhere to place P=+1 amplitudes.
    # Only neighbor present for P=0 is P=-1 and vice versa.
    source_ps = [state.sector.P for state in hybridization.states]
    assert source_ps == [-1, 0]
    delta = hybridization.dressed_eigenvectors_in_bare_basis - np.eye(
        2, dtype=np.complex128
    )
    assert not np.isclose(delta[0, 1], 0.0 + 0.0j)
    assert not np.isclose(delta[1, 0], 0.0 + 0.0j)


def test_rate_matrix_matches_fgr_helper_for_one_entry() -> None:
    model = _build_fixture(E_S=1e-3)
    hybridization = model.build_perturbative_hybridization_info()
    dressed_operator = model.build_dressed_operator_matrix(
        hybridization=hybridization,
        operator="n_operator",
    )

    def spectral_density(omega: float, temperature: float | None = None) -> float:
        return abs(omega) + (0.0 if temperature is None else 0.05 * temperature) + 0.3

    rates = model.compute_transition_rate_matrix_for_channel(
        hybridization=hybridization,
        channel=_channel_n("single", spectral_density),
        T=0.06,
    )

    i = 0
    j = 2
    expected = fgr_decay_rate(
        energy_i=float(hybridization.energies[i]),
        energy_j=float(hybridization.energies[j]),
        matrix_element=dressed_operator[i, j],
        spectral_density=spectral_density,
        T=0.06,
    )
    assert np.isclose(rates[i, j], expected, rtol=1e-12, atol=1e-12)


def test_multi_channel_total_decay_matches_channelwise_sum() -> None:
    model = _build_fixture(E_S=1e-3)

    def spectral_density_charge(omega: float, T: float | None = None) -> float:
        return abs(omega) + (0.0 if T is None else 0.2 * T) + 0.3

    def spectral_density_flux(omega: float, T: float | None = None) -> float:
        return 0.4 * abs(omega) + (0.0 if T is None else 0.1 * T) + 0.1

    channels = [
        _channel_n("charge", spectral_density_charge),
        _channel_n("flux", spectral_density_flux),
    ]
    per_channel, total = model.compute_multi_channel_decay_rates(
        channels=channels, T=0.06
    )
    assert set(per_channel) == {"charge", "flux"}
    all_keys = set(per_channel["charge"]) | set(per_channel["flux"]) | set(total)
    for key in all_keys:
        expected = per_channel["charge"].get(key, 0.0) + per_channel["flux"].get(
            key, 0.0
        )
        actual = total.get(key, 0.0)
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_multi_channel_decay_rates_include_total() -> None:
    model = _build_fixture(E_S=1e-3)

    def spectral_density_a(omega: float, T: float | None = None) -> float:
        return abs(omega) + (0.0 if T is None else T) + 0.2

    def spectral_density_b(omega: float, T: float | None = None) -> float:
        return 0.5 * abs(omega) + (0.0 if T is None else 0.25 * T) + 0.2

    channels = [
        _channel_n("channel_a", spectral_density_a),
        _channel_n("channel_b", spectral_density_b),
    ]
    per_channel, total = model.compute_multi_channel_decay_rates(
        channels=channels, T=0.05
    )
    assert set(per_channel) == {"channel_a", "channel_b"}
    assert len(total) > 0
    # total includes all positive entries from both channels
    assert set(total.keys()).issuperset(set(per_channel["channel_a"].keys()))
