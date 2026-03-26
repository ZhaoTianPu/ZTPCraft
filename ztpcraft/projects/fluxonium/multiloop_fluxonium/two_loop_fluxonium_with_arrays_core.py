# ==========================================
# Two-loop array fluxonium — CORE (nonlinear)
# ==========================================

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray
from typing import Any, cast, overload
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FloatArray = NDArray[np.float64]
ParamSet = dict[str, Any]

# Matplotlib typing stubs are incomplete; treat pyplot as dynamic.
plt = cast(Any, plt)


# -------- utilities --------
@overload
def wrap_coord(theta: float, wrap_width: float = 2.0 * np.pi) -> float: ...


@overload
def wrap_coord(theta: FloatArray, wrap_width: float = 2.0 * np.pi) -> FloatArray: ...


def wrap_coord(
    theta: float | FloatArray, wrap_width: float = 2.0 * np.pi
) -> float | FloatArray:
    wrapped = (
        np.asarray(theta, dtype=float) + wrap_width / 2.0
    ) % wrap_width - wrap_width / 2.0
    if np.isscalar(theta):
        return float(wrapped)
    return cast(FloatArray, wrapped)


def wrap_2pi(x: ArrayLike) -> FloatArray:
    return np.mod(np.asarray(x, dtype=np.float64), 2.0 * np.pi)


def _check_phi_shape(phi_array: Any, N: int) -> None:
    if len(phi_array) != 2 * N - 1:
        raise ValueError("phi_array must have length 2*N - 1.")


# -------- geometry --------
def symmetric_branch_flux(
    phi_array: Any,
    N: int,
    phi_ext_A: float,
    phi_ext_B: float,
    xp: Any = np,
) -> tuple[Any, Any, Any]:
    if N < 2:
        raise ValueError("N must be >= 2.")
    phi_array = xp.asarray(phi_array)

    phi_alpha_array = phi_array[: N - 1]
    phi_beta_array = phi_array[N - 1 : -1]
    phi_N = phi_array[-1]

    phi_branch_A_first = xp.asarray(
        [phi_alpha_array[0] - phi_ext_A / N], dtype=phi_array.dtype
    )
    phi_branch_B_first = xp.asarray(
        [phi_beta_array[0] + phi_ext_B / N], dtype=phi_array.dtype
    )

    if N > 2:
        phi_branch_A_mid = phi_alpha_array[1:] - phi_alpha_array[:-1] - phi_ext_A / N
        phi_branch_B_mid = phi_beta_array[1:] - phi_beta_array[:-1] + phi_ext_B / N
    else:
        phi_branch_A_mid = xp.asarray([], dtype=phi_array.dtype)
        phi_branch_B_mid = xp.asarray([], dtype=phi_array.dtype)

    phi_branch_A_last = xp.asarray(
        [phi_N - phi_alpha_array[N - 2] - phi_ext_A / N], dtype=phi_array.dtype
    )
    phi_branch_B_last = xp.asarray(
        [phi_N - phi_beta_array[N - 2] + phi_ext_B / N], dtype=phi_array.dtype
    )

    phi_branch_A = xp.concatenate(
        [phi_branch_A_first, phi_branch_A_mid, phi_branch_A_last]
    )
    phi_branch_B = xp.concatenate(
        [phi_branch_B_first, phi_branch_B_mid, phi_branch_B_last]
    )
    return phi_branch_A, phi_branch_B, -phi_N


def branch_flux_from_node_flux(
    node_flux: ArrayLike, param_set: ParamSet, phi_ext_A: float, phi_ext_B: float
) -> FloatArray:
    N_junctions_in_each_array = int(param_set["N"])
    _check_phi_shape(node_flux, N_junctions_in_each_array)
    branch_a, branch_b, phi_b = symmetric_branch_flux(
        np.asarray(node_flux, dtype=np.float64),
        N_junctions_in_each_array,
        phi_ext_A,
        phi_ext_B,
        xp=np,
    )
    return np.concatenate([branch_a, branch_b, [phi_b]])


def equilibrium_branch_flux_from_phi_b(
    phi_b: float,
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
    phase_slip_allocation_dict_list: list[dict[int, int]],
) -> FloatArray:
    N = int(param_set["N"])
    if not np.isclose(P_a, sum(phase_slip_allocation_dict_list[0].values())):
        raise ValueError("P_a does not match phase slip allocation for array A.")
    if not np.isclose(P_b, sum(phase_slip_allocation_dict_list[1].values())):
        raise ValueError("P_b does not match phase slip allocation for array B.")

    phi_a_tilde = (-phi_ext_A - P_a * 2 * np.pi - phi_b) / N
    phi_b_tilde = (phi_ext_B - P_b * 2 * np.pi - phi_b) / N
    phi_a_array = np.ones(N) * phi_a_tilde
    phi_b_array = np.ones(N) * phi_b_tilde
    for junction_idx, phase_slip_number in phase_slip_allocation_dict_list[0].items():
        phi_a_array[junction_idx] += phase_slip_number * 2 * np.pi
    for junction_idx, phase_slip_number in phase_slip_allocation_dict_list[1].items():
        phi_b_array[junction_idx] += phase_slip_number * 2 * np.pi
    return np.concatenate([phi_a_array, phi_b_array, [phi_b]])


# -------- potential --------
def potential_energy_node_flux(
    phi_array: Any,
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    xp: Any = np,
) -> Any:
    EJa = float(param_set["EJa"])
    EJb = float(param_set["EJb"])
    N = int(param_set["N"])
    _check_phi_shape(phi_array, N)

    phi_branch_A, phi_branch_B, phi_b = symmetric_branch_flux(
        phi_array, N, phi_ext_A, phi_ext_B, xp=xp
    )
    return -EJa * xp.sum(xp.cos(phi_branch_A) + xp.cos(phi_branch_B)) - EJb * xp.cos(
        phi_b
    )


def potential_energy_gradient_node_flux(
    phi_array: NDArray[np.float64],
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
) -> NDArray[np.float64]:
    EJa = float(param_set["EJa"])
    EJb = float(param_set["EJb"])
    N = int(param_set["N"])
    _check_phi_shape(phi_array, N)

    phi_branch_A, phi_branch_B, phi_b = symmetric_branch_flux(
        phi_array, N, phi_ext_A, phi_ext_B, xp=np
    )
    gradient_phi_alpha = EJa * (np.sin(phi_branch_A[:-1]) - np.sin(phi_branch_A[1:]))
    gradient_phi_beta = EJa * (np.sin(phi_branch_B[:-1]) - np.sin(phi_branch_B[1:]))
    gradient_phi_N = EJb * np.sin(phi_b) + EJa * (
        np.sin(phi_branch_A[-1]) + np.sin(phi_branch_B[-1])
    )
    return np.concatenate([gradient_phi_alpha, gradient_phi_beta, [gradient_phi_N]])


# -------- minima --------
def auxiliary_function_for_minima(
    phi_b: Any,
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
) -> Any:
    EJa = float(param_set["EJa"])
    EJb = float(param_set["EJb"])
    N = int(param_set["N"])
    cos_arg_1 = (phi_ext_B - phi_ext_A) / (2 * N) - phi_b / N - (P_a + P_b) * np.pi / N
    cos_arg_2 = (phi_ext_B + phi_ext_A) / (2 * N) - (P_b - P_a) * np.pi / N
    return -2 * EJa * N * np.cos(cos_arg_1) * np.cos(cos_arg_2) - EJb * np.cos(phi_b)


def auxiliary_function_first_derivative_for_minima(
    phi_b: Any,
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
) -> Any:
    EJa = float(param_set["EJa"])
    EJb = float(param_set["EJb"])
    N = int(param_set["N"])
    sin_arg_1 = (phi_ext_B - phi_ext_A) / (2 * N) - phi_b / N - (P_a + P_b) * np.pi / N
    cos_arg_2 = (phi_ext_B + phi_ext_A) / (2 * N) - (P_b - P_a) * np.pi / N
    return -2 * EJa * np.sin(sin_arg_1) * np.cos(cos_arg_2) + EJb * np.sin(phi_b)


def auxiliary_function_second_derivative_for_minima(
    phi_b: Any,
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
) -> Any:
    EJa = float(param_set["EJa"])
    EJb = float(param_set["EJb"])
    N = int(param_set["N"])
    cos_arg_1 = (phi_ext_B - phi_ext_A) / (2 * N) - phi_b / N - (P_a + P_b) * np.pi / N
    cos_arg_2 = (phi_ext_B + phi_ext_A) / (2 * N) - (P_b - P_a) * np.pi / N
    return 2 * EJa / N * np.cos(cos_arg_1) * np.cos(cos_arg_2) + EJb * np.cos(phi_b)


def node_flux_from_phi_b(
    phi_b: float, param_set: ParamSet, P_a: float, P_b: float
) -> FloatArray:
    N = int(param_set["N"])
    idx = np.arange(1, N, dtype=np.float64) / N
    phi_alpha_array = np.mod(idx * (-2.0 * np.pi * P_a - phi_b), 2.0 * np.pi)
    phi_beta_array = np.mod(idx * (-2.0 * np.pi * P_b - phi_b), 2.0 * np.pi)
    phi_N = np.mod(-phi_b, 2.0 * np.pi)
    return np.concatenate([phi_alpha_array, phi_beta_array, [phi_N]])


def bracket_roots(
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
    points_per_2pi: int = 50,
) -> list[tuple[float, float]]:
    N = int(param_set["N"])
    L = 2 * np.pi * N
    ngrid = int(points_per_2pi * N)
    phi = np.linspace(0, L, ngrid + 1)
    fvals = auxiliary_function_first_derivative_for_minima(
        phi, param_set, phi_ext_A, phi_ext_B, P_a, P_b
    )

    brackets: list[tuple[float, float]] = []
    for i in range(ngrid):
        f1 = fvals[i]
        f2 = fvals[i + 1]
        if np.isclose(f1, 0.0):
            brackets.append((float(phi[i]), float(phi[i])))
        elif f1 * f2 < 0:
            brackets.append((float(phi[i]), float(phi[i + 1])))
    return brackets


def _deduplicate_minima(
    phi_list: list[float],
    energy_list: list[float],
    tol: float = 1e-6,
) -> tuple[list[float], list[float]]:
    unique_phi: list[float] = []
    unique_energy: list[float] = []

    for phi, energy in zip(phi_list, energy_list):
        is_new = True
        for existing_phi in unique_phi:
            if abs(phi - existing_phi) < tol:
                is_new = False
                break

        if is_new:
            unique_phi.append(phi)
            unique_energy.append(energy)

    return unique_phi, unique_energy


def find_minima(
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
    bs_minima_threshold: float = 50,
) -> list[list[float]]:
    brackets = bracket_roots(param_set, phi_ext_A, phi_ext_B, P_a, P_b)
    N = int(param_set["N"])
    minima_Phi_b_list: list[float] = []
    minima_energy_list: list[float] = []

    for phi_left, phi_right in brackets:
        if phi_left == phi_right:
            x_opt = float(phi_left)
            f_opt = float(
                auxiliary_function_for_minima(
                    x_opt, param_set, phi_ext_A, phi_ext_B, P_a, P_b
                )
            )
        else:
            minima_result = sp.optimize.minimize_scalar(
                auxiliary_function_for_minima,
                args=(param_set, phi_ext_A, phi_ext_B, P_a, P_b),
                bounds=(phi_left, phi_right),
                method="bounded",
                options={"xatol": 1e-10},
            )
            x_opt = float(minima_result.x)
            f_opt = float(minima_result.fun)

        if (
            auxiliary_function_second_derivative_for_minima(
                x_opt, param_set, phi_ext_A, phi_ext_B, P_a, P_b
            )
            > 0
        ):
            minima_Phi_b_wrapped = float(wrap_coord(x_opt, 2 * np.pi * N))
            if abs(minima_Phi_b_wrapped) < bs_minima_threshold:
                minima_Phi_b_list.append(minima_Phi_b_wrapped)
                minima_energy_list.append(f_opt)

    # deduplicate
    minima_Phi_b_list, minima_energy_list = _deduplicate_minima(
        minima_Phi_b_list,
        minima_energy_list,
        tol=1e-6,
    )

    sorted_minima_energy = np.argsort(minima_energy_list)
    minima_phi_arr = np.array(minima_Phi_b_list)[sorted_minima_energy]
    minima_energy_arr = np.array(minima_energy_list)[sorted_minima_energy]
    return [
        [minima_Phi_b, minima_energy]
        for minima_Phi_b, minima_energy in zip(minima_phi_arr, minima_energy_arr)
    ]


## shelf for asymmetric homogeneous array


def auxiliary_function_for_minima_asymmetric_homogeneous_array(
    phi_b: float,
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
) -> float:
    EJa_alpha = float(param_set["EJa_alpha"])
    EJa_beta = float(param_set["EJa_beta"])
    EJb = float(param_set["EJb"])
    N_alpha = int(param_set["N_alpha"])
    N_beta = int(param_set["N_beta"])
    energy = 0.0
    energy += (
        -EJa_alpha
        * N_alpha
        * np.cos(-phi_ext_A / N_alpha - phi_b / N_alpha - P_a * 2 * np.pi / N_alpha)
    )
    energy += (
        -EJa_beta
        * N_beta
        * np.cos(phi_ext_B / N_beta - phi_b / N_beta - P_b * 2 * np.pi / N_beta)
    )
    energy += -EJb * np.cos(phi_b)
    return energy


def plot_auxiliary_potential_near_minima(
    param_set: ParamSet,
    phi_ext_A: float,
    phi_ext_B: float,
    P_a: float,
    P_b: float,
    minima: list[list[float]],
    window: float = 2 * np.pi,
    num_points: int = 400,
    show_quadratic: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot auxiliary potential U(phi_b) near each minimum.

    Args:
        minima: output of find_minima
        window: width of plotting region around each minimum
        show_quadratic: overlay quadratic approximation
    """

    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]

    for idx, (phi_b_min, energy_min) in enumerate(minima):
        # define local region
        phi_vals = np.linspace(
            phi_b_min - window / 2,
            phi_b_min + window / 2,
            num_points,
        )

        U_vals = auxiliary_function_for_minima(
            phi_vals,
            param_set,
            phi_ext_A,
            phi_ext_B,
            P_a,
            P_b,
        )

        ax.plot(phi_vals, U_vals, label=f"min {idx}")

        # mark minimum
        ax.scatter([phi_b_min], [energy_min], color="red")

        if show_quadratic:
            # second derivative at minimum
            curvature = auxiliary_function_second_derivative_for_minima(
                phi_b_min,
                param_set,
                phi_ext_A,
                phi_ext_B,
                P_a,
                P_b,
            )

            # quadratic approx
            delta = phi_vals - phi_b_min
            U_quad = energy_min + 0.5 * curvature * delta**2

            ax.plot(phi_vals, U_quad, linestyle="--", alpha=0.6)

    ax.set_xlabel(r"$\phi_b$")
    ax.set_ylabel("Auxiliary Potential")
    ax.set_title(f"Auxiliary potential near minima (P_a={P_a}, P_b={P_b})")
    ax.legend()
    ax.grid(True)

    return cast(Figure, fig), cast(Axes, ax)
