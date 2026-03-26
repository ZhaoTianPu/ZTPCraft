import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .gaussian_hermite_nd import G0_tensor, apply_L, integrate_gaussian_hermite
from .oscillators import LocalHarmonicOscillator

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OscillatorPairOverlapData:
    gaussian_quadratic: Array2D
    gaussian_linear: Array1D
    gaussian_constant: float
    hermite_directions: list[Array1D]
    hermite_centers: list[Array1D]
    num_modes_osc1: int
    num_modes_osc2: int
    transform_osc1: Array2D
    transform_osc2: Array2D
    freq_osc1: Array1D
    freq_osc2: Array1D
    center_osc1: Array1D
    center_osc2: Array1D


def prepare_oscillator_pair_overlap(
    transform_osc1: npt.ArrayLike,
    mode_freqs_osc1: npt.ArrayLike,
    center_osc1: npt.ArrayLike,
    transform_osc2: npt.ArrayLike,
    mode_freqs_osc2: npt.ArrayLike,
    center_osc2: npt.ArrayLike,
) -> OscillatorPairOverlapData:
    transform_osc1 = np.asarray(transform_osc1, dtype=float)
    transform_osc2 = np.asarray(transform_osc2, dtype=float)
    mode_freqs_osc1 = np.asarray(mode_freqs_osc1, dtype=float)
    mode_freqs_osc2 = np.asarray(mode_freqs_osc2, dtype=float)
    center_osc1 = np.asarray(center_osc1, dtype=float)
    center_osc2 = np.asarray(center_osc2, dtype=float)

    num_modes_osc1 = transform_osc1.shape[0]
    num_modes_osc2 = transform_osc2.shape[0]
    num_coordinates = transform_osc1.shape[1]

    assert transform_osc1.shape == (num_modes_osc1, num_coordinates)
    assert transform_osc2.shape == (num_modes_osc2, num_coordinates)
    assert mode_freqs_osc1.shape == (num_modes_osc1,)
    assert mode_freqs_osc2.shape == (num_modes_osc2,)
    assert center_osc1.shape == (num_coordinates,)
    assert center_osc2.shape == (num_coordinates,)

    quadratic_form_osc1: Array2D = np.asarray(
        transform_osc1.T @ np.diag(mode_freqs_osc1) @ transform_osc1, dtype=np.float64
    )
    quadratic_form_osc2: Array2D = np.asarray(
        transform_osc2.T @ np.diag(mode_freqs_osc2) @ transform_osc2, dtype=np.float64
    )

    gaussian_quadratic: Array2D = np.asarray(
        quadratic_form_osc1 + quadratic_form_osc2, dtype=np.float64
    )
    gaussian_linear: Array1D = np.asarray(
        -2.0 * (quadratic_form_osc1 @ center_osc1 + quadratic_form_osc2 @ center_osc2),
        dtype=np.float64,
    )
    gaussian_constant = float(
        center_osc1 @ quadratic_form_osc1 @ center_osc1
        + center_osc2 @ quadratic_form_osc2 @ center_osc2
    )

    hermite_directions: list[Array1D] = []
    hermite_centers: list[Array1D] = []

    # oscillator 1
    for mode_index in range(num_modes_osc1):
        hermite_direction = (
            np.sqrt(mode_freqs_osc1[mode_index]) * transform_osc1[mode_index]
        )
        hermite_directions.append(np.asarray(hermite_direction, dtype=float))
        hermite_centers.append(center_osc1.copy())

    # oscillator 2
    for mode_index in range(num_modes_osc2):
        hermite_direction = (
            np.sqrt(mode_freqs_osc2[mode_index]) * transform_osc2[mode_index]
        )
        hermite_directions.append(np.asarray(hermite_direction, dtype=float))
        hermite_centers.append(center_osc2.copy())

    return OscillatorPairOverlapData(
        gaussian_quadratic=gaussian_quadratic,
        gaussian_linear=gaussian_linear,
        gaussian_constant=gaussian_constant,
        hermite_directions=hermite_directions,
        hermite_centers=hermite_centers,
        num_modes_osc1=num_modes_osc1,
        num_modes_osc2=num_modes_osc2,
        transform_osc1=transform_osc1,
        transform_osc2=transform_osc2,
        freq_osc1=mode_freqs_osc1,
        freq_osc2=mode_freqs_osc2,
        center_osc1=center_osc1,
        center_osc2=center_osc2,
    )


@dataclass
class CachedOverlapInternals:
    mu: Array1D
    Gamma: Array2D
    M: Array2D
    Z0_exponent: float


def build_overlap_cache(
    pair_data: OscillatorPairOverlapData,
) -> CachedOverlapInternals:
    A: Array2D = pair_data.gaussian_quadratic
    B: Array1D = pair_data.gaussian_linear
    D_list: list[Array1D] = pair_data.hermite_directions

    L = np.linalg.cholesky(A)

    def solve_A(b: Array1D) -> Array1D:
        return np.asarray(np.linalg.solve(L.T, np.linalg.solve(L, b)), dtype=np.float64)

    mu: Array1D = np.asarray(-0.5 * solve_A(B), dtype=np.float64)

    logdetA = 2.0 * np.sum(np.log(np.diag(L)))
    BtAiB: float = float(B @ solve_A(B))

    Z0_exponent: float = float(
        A.shape[0] / 2 * np.log(2 * np.pi)
        + 0.125 * BtAiB
        - 0.5 * pair_data.gaussian_constant
        - 0.5 * logdetA
    )

    aux: Array2D = np.asarray(
        np.column_stack([solve_A(d) for d in D_list]), dtype=float
    )
    Gamma: Array2D = np.array(
        [
            [D_list[i] @ aux[:, j] for j in range(len(D_list))]
            for i in range(len(D_list))
        ],
        dtype=np.float64,
    )
    M: Array2D = np.asarray(2.0 * Gamma - np.eye(len(D_list)), dtype=np.float64)

    return CachedOverlapInternals(mu, Gamma, M, Z0_exponent)


def overlap_with_cache(
    pair_data: OscillatorPairOverlapData,
    cache: CachedOverlapInternals,
    occupations_osc1: Sequence[int],
    occupations_osc2: Sequence[int],
    *,
    use_logs: bool = False,
    report_prefactor_as_exponent: bool = False,
) -> complex | tuple[complex, float]:
    n1: npt.NDArray[np.int_] = np.asarray(occupations_osc1, dtype=int)
    n2: npt.NDArray[np.int_] = np.asarray(occupations_osc2, dtype=int)
    n_total: npt.NDArray[np.int_] = np.concatenate([n1, n2])

    D_list: list[Array1D] = pair_data.hermite_directions
    x0_list: list[Array1D] = pair_data.hermite_centers
    m = np.array([D @ (cache.mu - x0) for D, x0 in zip(D_list, x0_list)], dtype=complex)

    G0: npt.NDArray[np.complex128] = G0_tensor(m, n_total, use_logs)
    p_max = int(n_total.sum()) // 2
    g_tensor = G0
    accum = G0.copy()

    for p in range(1, p_max + 1):
        g_tensor = np.asarray(apply_L(g_tensor, cache.M), dtype=np.complex128) / p
        accum += g_tensor

    coeff = accum[tuple([-1] * len(n_total))]
    prefactor = sum(math.lgamma(int(n) + 1) for n in n_total)

    if np.isclose(coeff, 0.0, atol=1e-14, rtol=0.0):
        if report_prefactor_as_exponent:
            return complex(0.0, 0.0), -np.inf
        return 0.0 + 0.0j

    total_exponent = prefactor + cache.Z0_exponent
    if report_prefactor_as_exponent:
        return coeff, total_exponent
    return np.exp(np.log(coeff) + total_exponent)


def overlap_between_oscillator_fock_states(
    pair_data: OscillatorPairOverlapData,
    occupations_osc1: Sequence[int],
    occupations_osc2: Sequence[int],
    *,
    use_logs: bool = False,
    include_normalization: bool = False,
    report_prefactor_as_exponent: bool = False,
) -> complex | tuple[complex, complex]:
    occupations_osc1_arr: npt.NDArray[np.int_] = np.asarray(occupations_osc1, dtype=int)
    occupations_osc2_arr: npt.NDArray[np.int_] = np.asarray(occupations_osc2, dtype=int)

    assert occupations_osc1_arr.shape == (pair_data.num_modes_osc1,)
    assert occupations_osc2_arr.shape == (pair_data.num_modes_osc2,)
    assert np.all(occupations_osc1_arr >= 0)
    assert np.all(occupations_osc2_arr >= 0)

    if include_normalization:
        raise ValueError(
            "Normalization is handled in OscillatorOverlapEngine only. "
            "Pass include_normalization=False."
        )

    combined_occupations = np.concatenate([occupations_osc1_arr, occupations_osc2_arr])
    return integrate_gaussian_hermite(
        covariance_matrix=pair_data.gaussian_quadratic,
        linear_term=pair_data.gaussian_linear,
        constant_term=pair_data.gaussian_constant,
        hermite_directions=pair_data.hermite_directions,
        displacement_vectors=pair_data.hermite_centers,
        hermite_orders=combined_occupations,
        use_logs=use_logs,
        report_prefactor_as_exponent=report_prefactor_as_exponent,
    )


class OscillatorOverlapEngine:
    def __init__(
        self,
        osc1: LocalHarmonicOscillator,
        osc2: LocalHarmonicOscillator,
        *,
        use_cache: bool = True,
    ) -> None:
        self.osc1 = osc1
        self.osc2 = osc2
        self.pair_data = prepare_oscillator_pair_overlap(
            osc1.transform_xi_theta,
            osc1.frequencies,
            osc1.center,
            osc2.transform_xi_theta,
            osc2.frequencies,
            osc2.center,
        )
        self.cache: CachedOverlapInternals | None = (
            build_overlap_cache(self.pair_data) if use_cache else None
        )

    def overlap(
        self,
        n1: Sequence[int],
        n2: Sequence[int],
        *,
        use_logs: bool = False,
        include_normalization: bool = True,
        report_prefactor_as_exponent: bool = False,
    ) -> complex | tuple[complex, float | complex]:
        if self.cache is not None:
            val = overlap_with_cache(
                self.pair_data,
                self.cache,
                n1,
                n2,
                use_logs=use_logs,
                report_prefactor_as_exponent=report_prefactor_as_exponent,
            )
        else:
            val = overlap_between_oscillator_fock_states(
                self.pair_data,
                n1,
                n2,
                use_logs=use_logs,
                include_normalization=False,
                report_prefactor_as_exponent=report_prefactor_as_exponent,
            )

        if not include_normalization:
            return val

        if report_prefactor_as_exponent:
            assert isinstance(val, tuple)
            coeff, exponent = val
            return coeff, exponent + self._normalization(n1, n2)

        assert not isinstance(val, tuple)
        return val * self._normalization(n1, n2)

    def _normalization(self, n1: Sequence[int], n2: Sequence[int]) -> float:
        log_norm = 0.0

        # oscillator 1
        for freq, n in zip(self.osc1.frequencies, n1):
            log_norm += 0.25 * np.log(freq / np.pi)
            log_norm -= 0.5 * (int(n) * np.log(2.0) + math.lgamma(int(n) + 1))

        log_norm += 0.5 * self.osc1.log_jacobian()

        # oscillator 2
        for freq, n in zip(self.osc2.frequencies, n2):
            log_norm += 0.25 * np.log(freq / np.pi)
            log_norm -= 0.5 * (int(n) * np.log(2.0) + math.lgamma(int(n) + 1))

        log_norm += 0.5 * self.osc2.log_jacobian()

        return np.exp(log_norm)

    def batch(
        self,
        pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[complex | tuple[complex, float | complex]]:
        return [self.overlap(n1, n2) for n1, n2 in pairs]


def batched_overlaps(
    pair_data: OscillatorPairOverlapData,
    occupation_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    use_logs: bool = False,
    include_normalization: bool = False,
) -> list[complex | tuple[complex, complex]]:
    return [
        overlap_between_oscillator_fock_states(
            pair_data,
            occ1,
            occ2,
            use_logs=use_logs,
            include_normalization=include_normalization,
        )
        for occ1, occ2 in occupation_pairs
    ]


__all__ = [
    "OscillatorPairOverlapData",
    "prepare_oscillator_pair_overlap",
    "CachedOverlapInternals",
    "build_overlap_cache",
    "overlap_with_cache",
    "overlap_between_oscillator_fock_states",
    "OscillatorOverlapEngine",
    "batched_overlaps",
]
