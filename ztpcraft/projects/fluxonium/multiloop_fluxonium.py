import ztpcraft as ztpc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List
import scipy as sp
import scqubits as scq
from scipy.constants import Boltzmann, Planck
import copy

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Callable


## Fluxoid number model
def EL_fraction(EL_A, EL_B):
    EL_Sigma = EL_A + EL_B
    return EL_Sigma, EL_A / EL_Sigma, EL_B / EL_Sigma


def energy_offset_from_fluxoid_number(EL_A, EL_B, phi_ext_A, phi_ext_B, m_A, m_B):
    EL_Sigma, f_A, f_B = EL_fraction(EL_A, EL_B)
    return (
        0.5
        * EL_Sigma
        * (
            f_A * (phi_ext_A + 2 * np.pi * m_A) ** 2
            + f_B * (phi_ext_B + 2 * np.pi * m_B) ** 2
            - (
                f_A * (phi_ext_A + 2 * np.pi * m_A)
                - f_B * (phi_ext_B + 2 * np.pi * m_B)
            )
            ** 2
        )
    )


def single_mode_two_loop_fluxonium_potential_with_fluxoid_number(
    phi, EL_A, EL_B, EJb, phi_ext_A, phi_ext_B, m_A, m_B
):
    inductor_1_flux = phi - phi_ext_A - m_A * 2 * np.pi
    inductor_2_flux = phi + phi_ext_B + m_B * 2 * np.pi
    E_pot = (
        EL_A / 2 * (inductor_1_flux**2)
        + EL_B / 2 * (inductor_2_flux**2)
        - EJb * np.cos(phi)
    )
    return E_pot


def single_mode_flxn_potential_inductor_grouping(phi, EL, EJ, phi_ext):
    return 0.5 * EL * (phi + phi_ext) ** 2 - EJ * np.cos(phi)


def fluxoid_effective_ext_flux(f_A, f_B, phi_ext_A, phi_ext_B, m_A, m_B):
    return -f_A * (phi_ext_A + 2 * np.pi * m_A) + f_B * (phi_ext_B + 2 * np.pi * m_B)


def thermal_factor(freq, T):
    kbt = Boltzmann * T
    hf = Planck * freq
    return (1 / np.tanh(np.abs(hf) / (2 * kbt))) / (1 + np.exp(-(hf) / (kbt)))


def array_fluxonium_effective_ext_flux(P_a, P_b, phi_ext_a, phi_ext_b):
    phi_ext_effective = (phi_ext_b - phi_ext_a) / 2 - (P_a + P_b) * np.pi
    return phi_ext_effective


def energy_offset_from_array_fluxonium(param_set, phi_ext_a, phi_ext_b, P_a, P_b):
    EJa = param_set["EJa"]
    EL_A = EJa / param_set["N"]
    EL_B = EJa / param_set["N"]
    EL_Sigma, f_A, f_B = EL_fraction(EL_A, EL_B)
    return (
        0.5
        * EL_Sigma
        * (
            f_A * (phi_ext_a + 2 * np.pi * P_a) ** 2
            + f_B * (phi_ext_b - 2 * np.pi * P_b) ** 2
            - (
                f_A * (phi_ext_a + 2 * np.pi * P_a)
                - f_B * (phi_ext_b - 2 * np.pi * P_b)
            )
            ** 2
        )
    )


## NEB method
def _norm(v, eps=1e-12):
    return np.sqrt(np.dot(v, v) + eps)


def _unit(v, eps=1e-12):
    return v / _norm(v, eps)


def _interp_path(xA, xB, n_images):
    """Linear interpolation including endpoints."""
    xs = np.zeros((n_images, xA.size), dtype=float)
    for i in range(n_images):
        t = i / (n_images - 1)
        xs[i] = (1 - t) * xA + t * xB
    return xs


def _compute_tangents(images, energies, tangent="improved"):
    """
    Returns tangents tau_i for i=1..M-2 (endpoints unused).
    'improved' is the standard Henkelman-Jonsson tangent.
    """
    M, N = images.shape
    taus = np.zeros_like(images)

    for i in range(1, M - 1):
        d_plus = images[i + 1] - images[i]
        d_minus = images[i] - images[i - 1]

        if tangent == "simple":
            taus[i] = _unit(d_plus)
            continue

        Ei, Eip, Eim = energies[i], energies[i + 1], energies[i - 1]
        dE_plus = Eip - Ei
        dE_minus = Ei - Eim

        # Improved tangent: pick direction toward higher energy neighbor,
        # or a weighted mix if i is between a max and a min.
        if (Eip > Ei > Eim) or (Eip < Ei < Eim):
            # monotonic segment: choose the larger-energy direction
            taus[i] = _unit(d_plus if abs(dE_plus) >= abs(dE_minus) else d_minus)
        else:
            # i is at/near an extremum: weighted average
            w_plus = max(abs(dE_plus), abs(dE_minus))
            w_minus = min(abs(dE_plus), abs(dE_minus))
            if Eip > Eim:
                tvec = w_plus * d_plus + w_minus * d_minus
            else:
                tvec = w_plus * d_minus + w_minus * d_plus
            taus[i] = _unit(tvec)

    return taus


def neb(
    xA,
    xB,
    E,
    gradE,
    n_images=21,  # includes endpoints
    k_spring=1.0,
    step_size=0.05,
    max_steps=5000,
    force_tol=1e-5,
    climb=True,
    climb_start=100,  # when to enable climbing image
    reparam_every=10,  # redistribute images to avoid clustering
    tangent="improved",
    verbose=False,
):
    """
    Basic NEB with optional Climbing Image (CI-NEB).
    Endpoints fixed. Returns dict with images, energies, barrier, etc.
    """
    xA = np.asarray(xA, dtype=float)
    xB = np.asarray(xB, dtype=float)
    assert xA.shape == xB.shape
    N = xA.size
    M = int(n_images)
    assert M >= 3

    images = _interp_path(xA, xB, M)

    def redistribute(images):
        # Arc-length reparameterization (endpoints fixed)
        d = np.linalg.norm(np.diff(images, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        L = s[-1]
        if L < 1e-12:
            return images
        s_new = np.linspace(0.0, L, M)
        new = images.copy()
        j = 0
        for i in range(1, M - 1):
            while j < M - 2 and s[j + 1] < s_new[i]:
                j += 1
            # interpolate between j and j+1
            t = (s_new[i] - s[j]) / max(s[j + 1] - s[j], 1e-12)
            new[i] = (1 - t) * images[j] + t * images[j + 1]
        return new

    last_max_force = np.inf
    for it in range(max_steps):
        energies = np.array([E(img) for img in images], dtype=float)
        grads = np.array([gradE(img) for img in images], dtype=float)  # ∇E

        taus = _compute_tangents(images, energies, tangent=tangent)

        # Spring forces depend on neighbor distances along the band
        d_plus = images[2:] - images[1:-1]
        d_minus = images[1:-1] - images[:-2]
        l_plus = np.linalg.norm(d_plus, axis=1)
        l_minus = np.linalg.norm(d_minus, axis=1)

        # Index for climbing image: highest-energy *interior* image
        ci = None
        if climb and it >= climb_start:
            ci = 1 + np.argmax(energies[1:-1])

        max_force = 0.0

        # Update interior images
        for i in range(1, M - 1):
            tau = taus[i]

            g = grads[i]
            # True force is -∇E; NEB uses the perpendicular component only
            g_par = np.dot(g, tau) * tau
            g_perp = g - g_par

            # Spring force along tangent only
            f_spring = k_spring * (l_plus[i - 1] - l_minus[i - 1]) * tau

            # NEB "force" on image i
            F_neb = -g_perp + f_spring

            # Climbing image modification: remove spring, invert parallel component of true force
            if ci is not None and i == ci:
                # CI-NEB uses full true force with parallel inverted: F = -∇E + 2(∇E·tau)tau
                F_neb = -g + 2.0 * g_par

            max_force = max(max_force, _norm(F_neb))

            images[i] = images[i] + step_size * F_neb

        if reparam_every and (it + 1) % reparam_every == 0:
            images = redistribute(images)

        if verbose and (it % 50 == 0 or it == max_steps - 1):
            # Barrier estimate relative to min endpoint energy
            E0 = min(energies[0], energies[-1])
            barrier = float(np.max(energies) - E0)
            print(f"it={it:5d}  max|F|={max_force:.3e}  barrier~{barrier:.6g}")

        if max_force < force_tol:
            break

        # crude safeguard: if things blow up, reduce step
        if max_force > 10 * last_max_force and step_size > 1e-6:
            step_size *= 0.5
        last_max_force = max_force

    energies = np.array([E(img) for img in images], dtype=float)
    E0 = min(energies[0], energies[-1])
    barrier = float(np.max(energies) - E0)
    saddle_index = int(np.argmax(energies))

    return {
        "images": images,
        "energies": energies,
        "barrier": barrier,
        "saddle_index": saddle_index,
        "iterations": it + 1,
    }


def wrap_coord(theta, wrap_width=2 * np.pi):
    return (theta + wrap_width / 2) % wrap_width - wrap_width / 2


## Array junction phase slip model


# Helper to wrap to [0, 2π)
def wrap_2pi(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 2 * np.pi)


def potential_energy_node_flux(
    phi_array: NDArray[np.float64], param_set: dict, phi_ext_A: float, phi_ext_B: float
) -> float:
    """
    Calculate the potential energy of a two-loop fluxonium circuit.

    Args:
        phi_list (list): List of node flux values for the two-loop fluxonium circuit.
            The first N-1 values are the node fluxes for the alpha array, the next N-1 values are those for the beta array.
            The last value is the node flux for the end node.
        param_set (dict): Parameter set for the two-loop fluxonium circuit.

    Returns:
        float: Potential energy of the two-loop fluxonium circuit.
    """
    EJa = param_set["EJa"]
    EJb = param_set["EJb"]
    N = param_set["N"]
    assert len(phi_array) == 2 * N - 1

    phi_alpha_array = phi_array[: N - 1]
    phi_beta_array = phi_array[N - 1 : -1]
    phi_N = phi_array[-1]

    # calculate branch fluxes
    phi_branch_A = np.zeros(N)
    phi_branch_B = np.zeros(N)
    phi_branch_A[0] = phi_alpha_array[0] - phi_ext_A / N
    phi_branch_B[0] = phi_beta_array[0] + phi_ext_B / N
    for node_idx in range(1, N - 1):
        phi_branch_A[node_idx] = (
            phi_alpha_array[node_idx] - phi_alpha_array[node_idx - 1] - phi_ext_A / N
        )
        phi_branch_B[node_idx] = (
            phi_beta_array[node_idx] - phi_beta_array[node_idx - 1] + phi_ext_B / N
        )
    phi_branch_A[N - 1] = phi_N - phi_alpha_array[N - 2] - phi_ext_A / N
    phi_branch_B[N - 1] = phi_N - phi_beta_array[N - 2] + phi_ext_B / N
    phi_b = -phi_N

    # calculate potential energy
    E_pot = 0
    for branch_idx in range(N):
        E_pot += -EJa * (
            np.cos(phi_branch_A[branch_idx]) + np.cos(phi_branch_B[branch_idx])
        )
    E_pot += -EJb * np.cos(phi_b)
    return E_pot


def potential_energy_gradient_node_flux(
    phi_array: NDArray[np.float64], param_set: dict, phi_ext_A: float, phi_ext_B: float
) -> NDArray[np.float64]:
    """
    Calculate the gradient of the potential energy function with respect to the node fluxes.

    Args:
        phi_array (NDArray[np.float64]): Array of node fluxes.
        param_set (dict): Parameter set for the two-loop fluxonium circuit.
        phi_ext_A (float): External flux for the alpha array.
        phi_ext_B (float): External flux for the beta array.

    Returns:
        NDArray[np.float64]: Gradient of the potential energy function with respect to the node fluxes.
    """
    EJa = param_set["EJa"]
    EJb = param_set["EJb"]
    N = param_set["N"]
    assert len(phi_array) == 2 * N - 1

    phi_alpha_array = phi_array[: N - 1]
    phi_beta_array = phi_array[N - 1 : -1]
    phi_N = phi_array[-1]

    # calculate first-order derivatives
    gradient_phi_alpha = np.zeros(N - 1)
    gradient_phi_beta = np.zeros(N - 1)
    gradient_phi_N = 0

    # phi_alpha_1
    gradient_phi_alpha[0] = EJa * (
        np.sin(phi_alpha_array[0] - phi_ext_A / N)
        - np.sin(phi_alpha_array[1] - phi_alpha_array[0] - phi_ext_A / N)
    )
    # phi_alpha_N-1
    gradient_phi_alpha[N - 2] = EJa * (
        np.sin(phi_alpha_array[N - 2] - phi_alpha_array[N - 3] - phi_ext_A / N)
        - np.sin(phi_N - phi_alpha_array[N - 2] - phi_ext_A / N)
    )
    # phi_beta_1
    gradient_phi_beta[0] = EJa * (
        np.sin(phi_beta_array[0] + phi_ext_B / N)
        - np.sin(phi_beta_array[1] - phi_beta_array[0] + phi_ext_B / N)
    )
    # phi_beta_N-1
    gradient_phi_beta[N - 2] = EJa * (
        np.sin(phi_beta_array[N - 2] - phi_beta_array[N - 3] + phi_ext_B / N)
        - np.sin(phi_N - phi_beta_array[N - 2] + phi_ext_B / N)
    )
    for node_idx in range(1, N - 2):
        gradient_phi_alpha[node_idx] = EJa * (
            np.sin(
                phi_alpha_array[node_idx]
                - phi_alpha_array[node_idx - 1]
                - phi_ext_A / N
            )
            - np.sin(
                phi_alpha_array[node_idx + 1]
                - phi_alpha_array[node_idx]
                - phi_ext_A / N
            )
        )
        gradient_phi_beta[node_idx] = EJa * (
            np.sin(
                phi_beta_array[node_idx] - phi_beta_array[node_idx - 1] + phi_ext_B / N
            )
            - np.sin(
                phi_beta_array[node_idx + 1] - phi_beta_array[node_idx] + phi_ext_B / N
            )
        )

    # phi_N
    gradient_phi_N = -EJb * np.sin(-phi_N) + EJa * (
        np.sin(phi_N - phi_alpha_array[-1] - phi_ext_A / N)
        + np.sin(phi_N - phi_beta_array[-1] + phi_ext_B / N)
    )

    return np.concatenate([gradient_phi_alpha, gradient_phi_beta, [gradient_phi_N]])


def auxiliary_function_for_minima(phi_b, param_set, phi_ext_A, phi_ext_B, P_a, P_b):
    EJa = param_set["EJa"]
    EJb = param_set["EJb"]
    N = param_set["N"]
    cos_arg_1 = (phi_ext_B - phi_ext_A) / (2 * N) - phi_b / N - (P_a + P_b) * np.pi / N
    cos_arg_2 = (phi_ext_B + phi_ext_A) / (2 * N) - (P_b - P_a) * np.pi / N
    return -2 * EJa * N * np.cos(cos_arg_1) * np.cos(cos_arg_2) - EJb * np.cos(phi_b)


def auxiliary_function_for_minima_asymmetric_homogeneous_array(
    phi_b, param_set, phi_ext_A, phi_ext_B, P_a, P_b
):
    EJa_alpha = param_set["EJa_alpha"]
    EJa_beta = param_set["EJa_beta"]
    EJb = param_set["EJb"]
    N_alpha = param_set["N_alpha"]
    N_beta = param_set["N_beta"]
    energy = 0
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


def auxiliary_function_first_derivative_for_minima(
    phi_b, param_set, phi_ext_A, phi_ext_B, P_a, P_b
):
    EJa = param_set["EJa"]
    EJb = param_set["EJb"]
    N = param_set["N"]
    sin_arg_1 = (phi_ext_B - phi_ext_A) / (2 * N) - phi_b / N - (P_a + P_b) * np.pi / N
    cos_arg_2 = (phi_ext_B + phi_ext_A) / (2 * N) - (P_b - P_a) * np.pi / N
    return -2 * EJa * np.sin(sin_arg_1) * np.cos(cos_arg_2) + EJb * np.sin(phi_b)


def auxiliary_function_second_derivative_for_minima(
    phi_b, param_set, phi_ext_A, phi_ext_B, P_a, P_b
):
    EJa = param_set["EJa"]
    EJb = param_set["EJb"]
    N = param_set["N"]
    cos_arg_1 = (phi_ext_B - phi_ext_A) / (2 * N) - phi_b / N - (P_a + P_b) * np.pi / N
    cos_arg_2 = (phi_ext_B + phi_ext_A) / (2 * N) - (P_b - P_a) * np.pi / N
    return 2 * EJa / N * np.cos(cos_arg_1) * np.cos(cos_arg_2) + EJb * np.cos(phi_b)


def node_flux_from_phi_b(phi_b, param_set, P_a, P_b):
    N = param_set["N"]
    phi_alpha_array = np.zeros(N - 1)
    phi_beta_array = np.zeros(N - 1)
    phi_N = (-phi_b) % (2 * np.pi)
    for node_idx in range(N - 1):
        phi_alpha_array[node_idx] = (
            (node_idx + 1) / N * (-2 * np.pi * P_a - phi_b)
        ) % (2 * np.pi)
        phi_beta_array[node_idx] = ((node_idx + 1) / N * (-2 * np.pi * P_b - phi_b)) % (
            2 * np.pi
        )
    return np.concatenate([phi_alpha_array, phi_beta_array, [phi_N]])


def potential_energy_node_flux_jax(
    phi_array: jnp.ndarray,
    param_set: Dict,
    phi_ext_A: float,
    phi_ext_B: float,
) -> jnp.ndarray:
    """
    JAX version of the two-loop fluxonium potential energy.

    Args:
        phi_array: Array of node flux values, length 2*N - 1.
        param_set: Dict with keys "EJa", "EJb", "N".
        phi_ext_A: External flux in loop A (in radians of reduced flux).
        phi_ext_B: External flux in loop B (in radians of reduced flux).

    Returns:
        Scalar potential energy.
    """
    EJa = jnp.asarray(param_set["EJa"], dtype=jnp.float64)
    EJb = jnp.asarray(param_set["EJb"], dtype=jnp.float64)
    N = int(param_set["N"])  # treated as static Python int

    phi_array = jnp.asarray(phi_array, dtype=jnp.float64)

    # Unpack node fluxes
    phi_alpha_array = phi_array[: N - 1]
    phi_beta_array = phi_array[N - 1 : -1]
    phi_N = phi_array[-1]

    # Branch fluxes
    phi_branch_A = jnp.zeros((N,), dtype=jnp.float64)
    phi_branch_B = jnp.zeros((N,), dtype=jnp.float64)

    phi_branch_A = phi_branch_A.at[0].set(phi_alpha_array[0] - (phi_ext_A / N))
    phi_branch_B = phi_branch_B.at[0].set(phi_beta_array[0] + (phi_ext_B / N))

    if N > 2:
        delta_alpha = phi_alpha_array[1:] - phi_alpha_array[:-1] - (phi_ext_A / N)
        delta_beta = phi_beta_array[1:] - phi_beta_array[:-1] + (phi_ext_B / N)
        phi_branch_A = phi_branch_A.at[1 : N - 1].set(delta_alpha)
        phi_branch_B = phi_branch_B.at[1 : N - 1].set(delta_beta)

    phi_branch_A = phi_branch_A.at[N - 1].set(
        phi_N - phi_alpha_array[N - 2] - (phi_ext_A / N)
    )
    phi_branch_B = phi_branch_B.at[N - 1].set(
        phi_N - phi_beta_array[N - 2] + (phi_ext_B / N)
    )

    phi_b = -phi_N

    E_pot = -EJa * jnp.sum(jnp.cos(phi_branch_A) + jnp.cos(phi_branch_B))
    E_pot = E_pot - EJb * jnp.cos(phi_b)
    return E_pot


def potential_energy_asymmetric_homogeneous_array_node_flux_jax(
    phi_array: jnp.ndarray,
    param_set: Dict,
    phi_ext_A: float,
    phi_ext_B: float,
) -> jnp.ndarray:
    """
    JAX version of the potential energy function for an asymmetric homogeneous array of junctions.
    """
    EJa_alpha = jnp.asarray(param_set["EJa_alpha"], dtype=jnp.float64)
    EJa_beta = jnp.asarray(param_set["EJa_beta"], dtype=jnp.float64)
    EJb = jnp.asarray(param_set["EJb"], dtype=jnp.float64)
    N_alpha = int(param_set["N_alpha"])
    N_beta = int(param_set["N_beta"])

    assert len(phi_array) == N_alpha + N_beta - 1

    phi_alpha_array = phi_array[: N_alpha - 1]
    phi_beta_array = phi_array[N_alpha - 1 : -1]
    phi_N = phi_array[-1]

    # Branch fluxes
    phi_branch_A = jnp.zeros((N_alpha,), dtype=jnp.float64)
    phi_branch_B = jnp.zeros((N_beta,), dtype=jnp.float64)

    phi_branch_A = phi_branch_A.at[0].set(phi_alpha_array[0] - (phi_ext_A / N_alpha))
    phi_branch_B = phi_branch_B.at[0].set(phi_beta_array[0] + (phi_ext_B / N_beta))

    if N_alpha > 2:
        delta_alpha = phi_alpha_array[1:] - phi_alpha_array[:-1] - (phi_ext_A / N_alpha)
        delta_beta = phi_beta_array[1:] - phi_beta_array[:-1] + (phi_ext_B / N_beta)
        phi_branch_A = phi_branch_A.at[1 : N_alpha - 1].set(delta_alpha)
        phi_branch_B = phi_branch_B.at[1 : N_beta - 1].set(delta_beta)

    phi_branch_A = phi_branch_A.at[N_alpha - 1].set(
        phi_N - phi_alpha_array[N_alpha - 2] - (phi_ext_A / N_alpha)
    )
    phi_branch_B = phi_branch_B.at[N_beta - 1].set(
        phi_N - phi_beta_array[N_beta - 2] + (phi_ext_B / N_beta)
    )

    phi_b = -phi_N

    E_pot = -EJa_alpha * jnp.sum(jnp.cos(phi_branch_A)) - EJa_beta * jnp.sum(
        jnp.cos(phi_branch_B)
    )

    E_pot = E_pot - EJb * jnp.cos(phi_b)
    return E_pot


def potential_energy_asymmetric_homogeneous_array_node_flux(
    phi_array: np.ndarray,
    param_set: dict,
    phi_ext_A: float,
    phi_ext_B: float,
) -> np.ndarray:
    """
    JAX version of the potential energy function for an asymmetric homogeneous array of junctions.
    """
    EJa_alpha = np.asarray(param_set["EJa_alpha"], dtype=np.float64)
    EJa_beta = np.asarray(param_set["EJa_beta"], dtype=np.float64)
    EJb = np.asarray(param_set["EJb"], dtype=np.float64)
    N_alpha = int(param_set["N_alpha"])
    N_beta = int(param_set["N_beta"])

    assert len(phi_array) == N_alpha + N_beta - 1

    phi_alpha_array = phi_array[: N_alpha - 1]
    phi_beta_array = phi_array[N_alpha - 1 : -1]
    phi_N = phi_array[-1]

    # Branch fluxes
    phi_branch_A = np.zeros((N_alpha,), dtype=np.float64)
    phi_branch_B = np.zeros((N_beta,), dtype=np.float64)

    phi_branch_A[0] = phi_alpha_array[0] - (phi_ext_A / N_alpha)
    phi_branch_B[0] = phi_beta_array[0] + (phi_ext_B / N_beta)

    if N_alpha > 2:
        delta_alpha = phi_alpha_array[1:] - phi_alpha_array[:-1] - (phi_ext_A / N_alpha)
        delta_beta = phi_beta_array[1:] - phi_beta_array[:-1] + (phi_ext_B / N_beta)
        phi_branch_A[1 : N_alpha - 1] = delta_alpha
        phi_branch_B[1 : N_beta - 1] = delta_beta

    phi_branch_A[N_alpha - 1] = phi_N - phi_alpha_array[N_alpha - 2] - (phi_ext_A / N_alpha)
    phi_branch_B[N_beta - 1] = phi_N - phi_beta_array[N_beta - 2] + (phi_ext_B / N_beta)

    phi_b = -phi_N

    E_pot = -EJa_alpha * np.sum(np.cos(phi_branch_A)) - EJa_beta * np.sum(
        np.cos(phi_branch_B)
    )

    E_pot = E_pot - EJb * np.cos(phi_b)
    return E_pot


def inv_cap_matrix_asymmetric_homogeneous_array_node_flux(
    param_set: Dict,
    return_EC_matrix: bool = False,
) -> NDArray[np.float64]:
    """
    The kinetic matrix for an asymmetric homogeneous array of junctions.

    Capacitances need to be in fF.
    """
    N_alpha = int(param_set["N_alpha"])
    N_beta = int(param_set["N_beta"])
    CJa = param_set["CJa"]
    CJb = param_set["CJb"]
    Cga = param_set["Cga"]
    cap_matrix = np.zeros((N_alpha + N_beta - 1, N_alpha + N_beta - 1))

    alpha_array_node_idices = np.arange(N_alpha - 1)
    beta_array_node_idices = np.arange(N_alpha - 1, N_alpha + N_beta - 2)

    # diagonal terms
    for alpha_node_idx in alpha_array_node_idices:
        cap_matrix[alpha_node_idx, alpha_node_idx] += Cga + 2 * CJa
    for beta_node_idx in beta_array_node_idices:
        cap_matrix[beta_node_idx, beta_node_idx] += Cga + 2 * CJa
    cap_matrix[-1, -1] = 2 * CJa + CJb
    # off-diagonal terms
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
        inv_cap_matrix = ztpc.tb.units.capacitance_2_EC(1 / inv_cap_matrix)
    return inv_cap_matrix


def potential_energy_value_grad_hess(
    phi_array: jnp.ndarray,
    param_set: Dict,
    potential_energy_function: Callable,
    phi_ext_A: float,
    phi_ext_B: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convenience helper returning value, gradient, and Hessian w.r.t. phi_array.
    """
    energy_fn = lambda x: potential_energy_function(x, param_set, phi_ext_A, phi_ext_B)
    value, grad = jax.value_and_grad(energy_fn)(phi_array)
    hess = jax.jacfwd(jax.grad(energy_fn))(phi_array)
    return value, grad, hess


def transformation_matrix_branch_from_node(N_junctions_in_each_array):
    N_branches = 2 * N_junctions_in_each_array + 1
    N_node_in_array = N_junctions_in_each_array - 1
    N_nodes = 2 * (N_junctions_in_each_array - 1) + 1
    T = np.zeros((N_branches, N_branches))
    T[0, 0] = 1
    T[N_junctions_in_each_array - 1, N_node_in_array - 1] = -1
    T[N_junctions_in_each_array - 1, N_nodes - 1] = 1
    T[N_junctions_in_each_array, N_node_in_array] = 1
    T[N_branches - 2, N_nodes - 2] = -1
    T[N_branches - 2, N_nodes - 1] = 1
    T[N_branches - 1, N_nodes - 1] = -1
    for branch_idx in range(1, N_junctions_in_each_array - 1):
        T[branch_idx, branch_idx - 1] = -1
        T[branch_idx, branch_idx] = 1
        T[branch_idx + N_junctions_in_each_array, branch_idx - 1 + N_node_in_array] = -1
        T[branch_idx + N_junctions_in_each_array, branch_idx + N_node_in_array] = 1
    T[:N_junctions_in_each_array, -2] = -1 / N_junctions_in_each_array
    T[N_junctions_in_each_array:-1, -1] = 1 / N_junctions_in_each_array
    return T


def branch_flux_from_node_flux(node_flux, param_set, phi_ext_A, phi_ext_B):
    N_junctions_in_each_array = param_set["N"]
    T = transformation_matrix_branch_from_node(N_junctions_in_each_array)
    augmented_node_flux = np.concatenate([node_flux, [phi_ext_A, phi_ext_B]])
    branch_flux = T @ augmented_node_flux
    return branch_flux


def equilibrium_branch_flux_from_phi_b(
    phi_b, param_set, phi_ext_A, phi_ext_B, P_a, P_b, phase_slip_allocation_dict_list
):
    N = param_set["N"]
    assert np.isclose(P_a, sum(phase_slip_allocation_dict_list[0].values()))
    assert np.isclose(P_b, sum(phase_slip_allocation_dict_list[1].values()))
    Phi_a_tilde = (-phi_ext_A - P_a * 2 * np.pi - phi_b) / N
    Phi_b_tilde = (phi_ext_B - P_b * 2 * np.pi - phi_b) / N
    Phi_a_array = np.ones(N) * Phi_a_tilde
    Phi_b_array = np.ones(N) * Phi_b_tilde
    for junction_idx, phase_slip_number in phase_slip_allocation_dict_list[0].items():
        Phi_a_array[junction_idx] += phase_slip_number * 2 * np.pi
    for junction_idx, phase_slip_number in phase_slip_allocation_dict_list[1].items():
        Phi_b_array[junction_idx] += phase_slip_number * 2 * np.pi
    branch_flux = np.concatenate([Phi_a_array, Phi_b_array, [phi_b]])
    return branch_flux


def single_mode_two_loop_fluxonium_with_fluxoid_number(
    phi, param_set, phi_ext_A, phi_ext_B, m_a, m_b
):
    N = param_set["N"]
    EJa = param_set["EJa"]
    inductor_1_flux = phi - phi_ext_A - m_a * 2 * np.pi
    inductor_2_flux = phi + phi_ext_B + m_b * 2 * np.pi
    EL_per_branch = EJa / N
    EJb = param_set["EJb"]
    E_pot = EL_per_branch / 2 * (
        inductor_1_flux**2 + inductor_2_flux**2
    ) - EJb * np.cos(phi)
    return E_pot


from typing import Any


def bracket_roots(param_set, phi_ext_A, phi_ext_B, P_a, P_b, points_per_2pi=50):
    """
    Return list of (phi_left, phi_right) brackets
    covering all roots in [0, 2πN).
    """
    N = param_set["N"]
    L = 2 * np.pi * N
    ngrid = int(points_per_2pi * N)

    phi = np.linspace(0, L, ngrid + 1)
    fvals = auxiliary_function_first_derivative_for_minima(
        phi, param_set, phi_ext_A, phi_ext_B, P_a, P_b
    )

    brackets = []

    for i in range(ngrid):
        f1 = fvals[i]
        f2 = fvals[i + 1]

        # exact zero (rare but handle it)
        if f1 == 0.0:
            brackets.append((phi[i], phi[i]))
        # sign change
        elif f1 * f2 < 0:
            brackets.append((phi[i], phi[i + 1]))

    return brackets


def find_minima(param_set, phi_ext_A, phi_ext_B, P_a, P_b, bs_minima_threshold=50):
    brackets = bracket_roots(param_set, phi_ext_A, phi_ext_B, P_a, P_b)
    N = param_set["N"]
    minima_Phi_b_list = []
    minima_energy_list = []

    for min, max in brackets:
        minima_result = sp.optimize.minimize(
            auxiliary_function_for_minima,
            min,
            args=(param_set, phi_ext_A, phi_ext_B, P_a, P_b),
            bounds=[(min, max)],
            method="Nelder-Mead",
            tol=1e-10,
        )
        if (
            auxiliary_function_second_derivative_for_minima(
                minima_result.x[0], param_set, phi_ext_A, phi_ext_B, P_a, P_b
            )
            > 0
        ):
            minima_Phi_b_wrapped = wrap_coord(minima_result.x[0], 2 * np.pi * N)
            if abs(minima_Phi_b_wrapped) < bs_minima_threshold:
                minima_Phi_b_list.append(minima_Phi_b_wrapped)
                minima_energy_list.append(minima_result.fun)

    # sort minima by energy
    sorted_minima_energy = np.argsort(minima_energy_list)
    minima_Phi_b_list = np.array(minima_Phi_b_list)[sorted_minima_energy]
    minima_energy_list = np.array(minima_energy_list)[sorted_minima_energy]
    return [
        [minima_Phi_b, minima_energy]
        for minima_Phi_b, minima_energy in zip(minima_Phi_b_list, minima_energy_list)
    ]


## Normal modes

from scipy.sparse.linalg import eigsh


def sym_matrix_sqrt(A, rtol=1e-12, atol=0.0):
    # A: real symmetric (assumed)
    w, Q = np.linalg.eigh(A)
    # threshold for numerical zeros
    thresh = np.max([atol, rtol * np.max(np.abs(w))])
    w_pos = np.clip(w, 0.0, None)  # avoid negative due to roundoff
    if np.any(w < -10 * thresh):
        raise ValueError("A is not SPD/PSD: has significantly negative eigenvalues.")
    s = np.sqrt(np.where(w_pos > thresh, w_pos, 0.0))
    Ahalf = (Q * s) @ Q.T
    # force symmetry
    return 0.5 * (Ahalf + Ahalf.T)


def sym_matrix_inv_sqrt(A, rtol=1e-12, atol=0.0, pseudo=True):
    w, Q = np.linalg.eigh(A)
    thresh = np.max([atol, rtol * np.max(np.abs(w))])
    if np.any(w < -10 * thresh):
        raise ValueError("A is not SPD/PSD: has significantly negative eigenvalues.")
    # Build Λ^{-1/2}
    inv_s = np.zeros_like(w)
    mask = w > thresh
    inv_s[mask] = 1.0 / np.sqrt(w[mask])
    if not pseudo and not np.all(mask):
        raise ValueError(
            "A is singular/near-singular; set pseudo=True for pseudoinverse."
        )
    Ainvhalf = (Q * inv_s) @ Q.T
    return 0.5 * (Ainvhalf + Ainvhalf.T)


def normal_modes_canonical(EC_mat, hessian_mat, *, atol=1e-12):
    """
    Diagonalize H = 4 n^T EC_mat n + 1/2 theta^T hessian_mat theta
    EC_mat_prime = EC_mat*8 so that
    H = 1/2 n^T EC_mat_prime n + 1/2 theta^T hessian_mat theta
    assuming EC_mat is symmetric positive definite and hessian_mat symmetric.

    Returns
    -------
    mode_freqs : (n,) ndarray
        Mode frequencies (nonnegative).
    S_theta : (n,n) ndarray
        Columns are mode shapes in theta-space: theta = S_theta @ Q  (so Q = S_theta.T @ theta? see below).
        Here we use the canonical choice Q = U^T EC_mat_prime^{1/2} x, hence S_theta = EC_mat_prime^{-1/2} U.
    S_n : (n,n) ndarray
        charge operator map columns: n = S_n @ Q with Q = U^T EC_mat_prime^{1/2} n, so S_n = EC_mat_prime^{-1/2} U.
        (You usually don't need S_n explicitly, but it's handy to verify canonicity.)
    U  : (n,n) ndarray
        Orthogonal eigenvectors of C = EC_mat_prime^{1/2} hessian_mat EC_mat_prime^{1/2}.
    """
    # Symmetrize inputs numerically
    EC_mat = 0.5 * (EC_mat + EC_mat.T)
    EC_mat_prime = EC_mat * 8
    hessian_mat = 0.5 * (hessian_mat + hessian_mat.T)

    # get square root of EC_mat_prime
    EC_mat_prime_onehalf = sym_matrix_sqrt(EC_mat_prime)
    EC_mat_prime_minus_onehalf = sym_matrix_inv_sqrt(EC_mat_prime)

    # Build symmetric C = EC_mat_prime^{1/2} hessian_mat EC_mat_prime^{1/2}
    C = EC_mat_prime_onehalf @ hessian_mat @ EC_mat_prime_onehalf

    # Diagonalize C (symmetric)
    lam, U = np.linalg.eigh(C)  # lam = ω^2, U orthogonal

    # # Clean tiny negatives due to roundoff
    # lam[lam < 0] = np.where(lam < -atol, lam, 0.0)
    mode_freqs = np.sqrt(lam.clip(min=0.0))

    # Maps between (theta, p) and normal (xi, q):
    # q = U^T EC_mat_prime^{1/2} p  ⇒ p = EC_mat_prime^{-1/2} U q
    # xi = U^T EC_mat_prime^{-1/2} theta ⇒ theta = EC_mat_prime^{1/2} U xi
    S_theta_xi = EC_mat_prime_onehalf @ U  # theta = S_theta_xi @ xi
    S_p_q = EC_mat_prime_minus_onehalf @ U  # p = S_p_q @ q

    # Zero-point fluctuations in normal coordinates
    # Guard ω=0 (free/Goldstone): set rms to +inf to signal divergence
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_zpf_modes = np.sqrt(1 / (2.0 * mode_freqs))
        q_zpf_modes = np.sqrt(mode_freqs / 2.0)
        xi_zpf_modes = np.where(mode_freqs > atol, xi_zpf_modes, np.inf)
        q_zpf_modes = np.where(mode_freqs > atol, q_zpf_modes, 0.0)  # ω→0 ⇒ P rms → 0
        xi_osc_length = np.sqrt(1 / mode_freqs)

    return mode_freqs, S_theta_xi, S_p_q, U, xi_zpf_modes, q_zpf_modes, xi_osc_length


def normal_modes_gen_eval(EC_mat, hessian_mat, *, atol=1e-12):
    """
    Diagonalize H = 4 n^T EC_mat n + 1/2 theta^T hessian_mat theta
    EC_mat_prime = EC_mat*8 so that
    H = 1/2 n^T EC_mat_prime n + 1/2 theta^T hessian_mat theta
    assuming EC_mat is symmetric positive definite and hessian_mat symmetric.

    Returns
    -------
    mode_freqs : (n,) ndarray
        Mode frequencies (nonnegative).
    S_theta : (n,n) ndarray
        Columns are mode shapes in theta-space: theta = S_theta @ Q  (so Q = S_theta.T @ theta? see below).
        Here we use the canonical choice Q = U^T EC_mat_prime^{1/2} x, hence S_theta = EC_mat_prime^{-1/2} U.
    S_n : (n,n) ndarray
        charge operator map columns: n = S_n @ Q with Q = U^T EC_mat_prime^{1/2} n, so S_n = EC_mat_prime^{-1/2} U.
        (You usually don't need S_n explicitly, but it's handy to verify canonicity.)
    U  : (n,n) ndarray
        Orthogonal eigenvectors of C = EC_mat_prime^{1/2} hessian_mat EC_mat_prime^{1/2}.
    """
    # Symmetrize inputs numerically
    EC_mat = 0.5 * (EC_mat + EC_mat.T)
    EC_mat_prime = EC_mat * 8
    hessian_mat = 0.5 * (hessian_mat + hessian_mat.T)

    mode_freqs_sq, S_theta_xi = sp.linalg.eigh(hessian_mat, np.linalg.inv(EC_mat_prime))
    mode_freqs = np.sqrt(mode_freqs_sq)
    # Zero-point fluctuations in normal coordinates
    # Guard ω=0 (free/Goldstone): set rms to +inf to signal divergence
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_zpf_modes = np.sqrt(1 / (2.0 * mode_freqs))
        q_zpf_modes = np.sqrt(mode_freqs / 2.0)
        xi_zpf_modes = np.where(mode_freqs > atol, xi_zpf_modes, np.inf)
        q_zpf_modes = np.where(mode_freqs > atol, q_zpf_modes, 0.0)  # ω→0 ⇒ P rms → 0
        xi_osc_length = np.sqrt(1 / mode_freqs)

    return mode_freqs, S_theta_xi, xi_zpf_modes, q_zpf_modes, xi_osc_length
