# ==========================================
# Two-loop array fluxonium — LINEARIZED
# ==========================================

import numpy as np
import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from typing import Any, cast

import ztpcraft as ztpc

from ztpcraft.bosonic.oscillator_integrals.normal_modes import (
    diagonalize_quadratic_hamiltonian,
)

from .two_loop_fluxonium_with_arrays_core import (
    find_minima,
    node_flux_from_phi_b,
    potential_energy_node_flux,
)

from ztpcraft.bosonic.oscillator_integrals.oscillators import (
    LocalHarmonicOscillator,
    oscillator_from_circuit,
)

from ztpcraft.bosonic.oscillator_integrals.oscillator_system import (
    OscillatorRegistry,
    OscillatorSystem,
)


from .two_loop_fluxonium_with_arrays_core import plot_auxiliary_potential_near_minima

ParamSet = dict[str, Any]
FloatArray = NDArray[np.float64]

from dataclasses import dataclass

# Use double precision for stable gradient/Hessian checks at minima.
jax.config.update("jax_enable_x64", True)  # type: ignore[reportUnknownMemberType]


@dataclass
class FluxoniumMinimum:
    name: str
    P_a: float
    P_b: float
    phi_b: float
    phi_eq: FloatArray

    energy_classical: float

    oscillator: LocalHarmonicOscillator


def inv_cap_matrix_homogeneous_array_node_flux(
    param_set: ParamSet,
    return_EC_matrix: bool = False,
) -> NDArray[np.float64]:
    """
    The kinetic matrix for an asymmetric homogeneous array of junctions.

    Capacitances need to be in fF.
    """
    if "N" in param_set and ("N_alpha" not in param_set or "N_beta" not in param_set):
        N_alpha = int(param_set["N"])
        N_beta = int(param_set["N"])
    elif "N_alpha" in param_set and "N_beta" in param_set and "N" not in param_set:
        N_alpha = int(param_set["N_alpha"])
        N_beta = int(param_set["N_beta"])
    else:
        raise ValueError(
            "Either N or N_alpha and N_beta must be provided in the param_set."
        )
    CJa = float(param_set["CJa"])
    CJb = float(param_set["CJb"])
    Cga = float(param_set["Cga"])
    cap_matrix = np.zeros((N_alpha + N_beta - 1, N_alpha + N_beta - 1))

    alpha_array_node_idices = np.arange(N_alpha - 1)
    beta_array_node_idices = np.arange(N_alpha - 1, N_alpha + N_beta - 2)

    for alpha_node_idx in alpha_array_node_idices:
        cap_matrix[alpha_node_idx, alpha_node_idx] += Cga + 2 * CJa
    for beta_node_idx in beta_array_node_idices:
        cap_matrix[beta_node_idx, beta_node_idx] += Cga + 2 * CJa
    cap_matrix[-1, -1] = 2 * CJa + CJb

    for alpha_node_idx in alpha_array_node_idices[1:]:
        cap_matrix[alpha_node_idx, alpha_node_idx - 1] = -CJa
        cap_matrix[alpha_node_idx - 1, alpha_node_idx] = -CJa
    for beta_node_idx in beta_array_node_idices[1:]:
        cap_matrix[beta_node_idx, beta_node_idx - 1] = -CJa
        cap_matrix[beta_node_idx - 1, beta_node_idx] = -CJa
    cap_matrix[alpha_array_node_idices[-1], -1] = -CJa
    cap_matrix[-1, alpha_array_node_idices[-1]] = -CJa
    cap_matrix[beta_array_node_idices[-1], -1] = -CJa
    cap_matrix[-1, beta_array_node_idices[-1]] = -CJa

    inv_cap_matrix = np.linalg.inv(cap_matrix)
    if return_EC_matrix:
        inv_cap_matrix = cast(
            NDArray[np.float64],
            ztpc.tb.units.capacitance_2_EC(1 / inv_cap_matrix),  # type: ignore[reportUnknownMemberType]
        )
    return cast(NDArray[np.float64], inv_cap_matrix)


def potential_value_grad_hessian(
    phi_array: Any, param_set: ParamSet, phi_ext_A: float, phi_ext_B: float
) -> tuple[Any, Any, Any]:
    def energy_fn(x: jnp.ndarray) -> Any:
        return potential_energy_node_flux(x, param_set, phi_ext_A, phi_ext_B, xp=jnp)

    value, grad = jax.value_and_grad(energy_fn)(phi_array)  # type: ignore[reportUnknownVariableType]
    hess = cast(
        Any,
        jax.jacfwd(jax.grad(energy_fn))(phi_array),  # type: ignore[reportUnknownVariableType]
    )
    return value, grad, hess


def normal_modes_from_matrices(
    EC_mat: FloatArray, hessian_mat: FloatArray, atol: float = 1e-12
):
    return diagonalize_quadratic_hamiltonian(EC_mat, hessian_mat, atol=atol)


class TwoLoopArrayFluxonium:
    def __init__(self, param_set: ParamSet):
        self.params = param_set
        self.minima_data: dict[str, FluxoniumMinimum] = {}

    def get_minimum(self, name: str) -> FluxoniumMinimum:
        return self.minima_data[name]

    # ---------- core physics ----------
    def potential(self, phi: Any, phi_ext_A: float, phi_ext_B: float) -> Any:
        return potential_energy_node_flux(phi, self.params, phi_ext_A, phi_ext_B)

    # ---------- expansion ----------
    def value_grad_hessian(
        self, phi: Any, phi_ext_A: float, phi_ext_B: float
    ) -> tuple[Any, Any, Any]:
        return potential_value_grad_hessian(phi, self.params, phi_ext_A, phi_ext_B)

    # ---------- capacitance ----------
    def capacitance_matrix(self) -> NDArray[np.float64]:
        return inv_cap_matrix_homogeneous_array_node_flux(
            self.params, return_EC_matrix=True
        )

    # ---------- normal modes ----------
    def normal_modes(
        self,
        phi_eq: jnp.ndarray,
        phi_ext_A: float,
        phi_ext_B: float,
        atol: float = 1e-12,
    ):
        _, _, hess = self.value_grad_hessian(phi_eq, phi_ext_A, phi_ext_B)
        EC = self.capacitance_matrix()
        return diagonalize_quadratic_hamiltonian(EC, np.asarray(hess), atol=atol)

    def minima_for_sector(
        self,
        phi_ext_A: float,
        phi_ext_B: float,
        P_a: float,
        P_b: float,
    ) -> list[list[float]]:
        return find_minima(
            self.params,
            phi_ext_A,
            phi_ext_B,
            P_a,
            P_b,
        )

    def oscillator_from_minimum(
        self,
        phi_b: float,
        phi_ext_A: float,
        phi_ext_B: float,
        P_a: float,
        P_b: float,
    ) -> LocalHarmonicOscillator:
        # 1. get node flux center
        phi_eq: FloatArray = node_flux_from_phi_b(phi_b, self.params, P_a, P_b)

        # 2. compute Hessian
        _, _, hess_any = self.value_grad_hessian(
            phi_eq,
            phi_ext_A,
            phi_ext_B,
        )
        hess = np.asarray(hess_any, dtype=np.float64)

        # 3. capacitance matrix
        EC = self.capacitance_matrix()

        # 4. build oscillator
        return oscillator_from_circuit(
            EC,
            np.asarray(hess),
            theta_center=np.asarray(phi_eq),
        )

    def oscillators_for_sector(
        self,
        phi_ext_A: float,
        phi_ext_B: float,
        P_a: float,
        P_b: float,
        name_format: str = "Pa_{P_a}_Pb_{P_b}_minidx_{minidx}",
    ) -> list[FluxoniumMinimum]:
        minima = self.minima_for_sector(phi_ext_A, phi_ext_B, P_a, P_b)

        results: list[FluxoniumMinimum] = []
        for idx, (phi_b, energy) in enumerate(minima):
            phi_eq = node_flux_from_phi_b(phi_b, self.params, P_a, P_b)

            osc = self.oscillator_from_minimum(
                float(phi_b),
                phi_ext_A,
                phi_ext_B,
                P_a,
                P_b,
            )

            name = name_format.format(P_a=int(P_a), P_b=int(P_b), minidx=int(idx))
            minimum = FluxoniumMinimum(
                name=name,
                P_a=P_a,
                P_b=P_b,
                phi_b=phi_b,
                phi_eq=np.asarray(phi_eq),
                energy_classical=energy,
                oscillator=osc,
            )
            self.minima_data[name] = minimum
            results.append(minimum)

        return results

    def build_registry(
        self,
        phi_ext_A: float,
        phi_ext_B: float,
        sectors: list[tuple[float, float]],
        name_format: str = "Pa_{P_a}_Pb_{P_b}_minidx_{minidx}",
    ):
        registry = OscillatorRegistry()

        for P_a, P_b in sectors:
            minima: list[FluxoniumMinimum] = self.oscillators_for_sector(
                phi_ext_A,
                phi_ext_B,
                P_a,
                P_b,
                name_format=name_format,
            )

            for minimum in minima:
                registry.add_minimum(minimum.name, minimum.oscillator)

        return registry

    def build_system(
        self,
        phi_ext_A: float,
        phi_ext_B: float,
        sectors: list[tuple[float, float]],
    ):
        registry = self.build_registry(phi_ext_A, phi_ext_B, sectors)
        return OscillatorSystem(registry)

    def plot_auxiliary_near_minima(
        self,
        phi_ext_A: float,
        phi_ext_B: float,
        P_a: float,
        P_b: float,
        window: float = 2 * np.pi,
        num_points: int = 400,
        show_quadratic: bool = True,
    ):
        minima = self.minima_for_sector(phi_ext_A, phi_ext_B, P_a, P_b)

        return plot_auxiliary_potential_near_minima(
            self.params,
            phi_ext_A,
            phi_ext_B,
            P_a,
            P_b,
            minima,
            window=window,
            num_points=num_points,
            show_quadratic=show_quadratic,
        )
