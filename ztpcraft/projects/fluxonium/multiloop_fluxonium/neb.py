"""Nudged Elastic Band (NEB) utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class NebResult(TypedDict):
    images: FloatArray
    energies: FloatArray
    barrier: float
    saddle_index: int
    iterations: int


def _norm(v: FloatArray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.dot(v, v) + eps))


def _unit(v: FloatArray, eps: float = 1e-12) -> FloatArray:
    return cast(FloatArray, v / _norm(v, eps))


def _interp_path(xA: FloatArray, xB: FloatArray, n_images: int) -> FloatArray:
    """Linear interpolation including endpoints."""
    xs = np.zeros((n_images, xA.size), dtype=float)
    for i in range(n_images):
        t = i / (n_images - 1)
        xs[i] = (1.0 - t) * xA + t * xB
    return xs


def _compute_tangents(
    images: FloatArray,
    energies: FloatArray,
    tangent: Literal["improved", "simple"] = "improved",
) -> FloatArray:
    """Return tangents tau_i for i=1..M-2; endpoints are unused."""
    m_count, _ = images.shape
    taus = np.zeros_like(images)

    for i in range(1, m_count - 1):
        d_plus = images[i + 1] - images[i]
        d_minus = images[i] - images[i - 1]

        if tangent == "simple":
            taus[i] = _unit(d_plus)
            continue

        ei = float(energies[i])
        eip = float(energies[i + 1])
        eim = float(energies[i - 1])
        dE_plus = eip - ei
        dE_minus = ei - eim

        if (eip > ei > eim) or (eip < ei < eim):
            taus[i] = _unit(d_plus if abs(dE_plus) >= abs(dE_minus) else d_minus)
        else:
            w_plus = max(abs(dE_plus), abs(dE_minus))
            w_minus = min(abs(dE_plus), abs(dE_minus))
            tvec = (
                w_plus * d_plus + w_minus * d_minus
                if eip > eim
                else w_plus * d_minus + w_minus * d_plus
            )
            taus[i] = _unit(tvec)

    return taus


def neb(
    xA: ArrayLike,
    xB: ArrayLike,
    E: Callable[[FloatArray], float],
    gradE: Callable[[FloatArray], ArrayLike],
    n_images: int = 21,
    k_spring: float = 1.0,
    step_size: float = 0.05,
    max_steps: int = 5000,
    force_tol: float = 1e-5,
    climb: bool = True,
    climb_start: int = 100,
    reparam_every: int = 10,
    tangent: Literal["improved", "simple"] = "improved",
    verbose: bool = False,
) -> NebResult:
    """Basic NEB with optional Climbing Image (CI-NEB)."""
    xA_arr = np.asarray(xA, dtype=float)
    xB_arr = np.asarray(xB, dtype=float)
    if xA_arr.shape != xB_arr.shape:
        raise ValueError("xA and xB must have the same shape.")

    m_count = int(n_images)
    if m_count < 3:
        raise ValueError("n_images must be >= 3.")

    images = _interp_path(xA_arr, xB_arr, m_count)

    def redistribute(path_images: FloatArray) -> FloatArray:
        d = np.linalg.norm(np.diff(path_images, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        total_len = float(s[-1])
        if total_len < 1e-12:
            return path_images
        s_new = np.linspace(0.0, total_len, m_count)
        new = path_images.copy()
        j = 0
        for i in range(1, m_count - 1):
            while j < m_count - 2 and s[j + 1] < s_new[i]:
                j += 1
            t = (s_new[i] - s[j]) / max(float(s[j + 1] - s[j]), 1e-12)
            new[i] = (1.0 - t) * path_images[j] + t * path_images[j + 1]
        return new

    last_max_force = float(np.inf)
    iterations = 0
    for it in range(max_steps):
        iterations = it + 1
        energies = np.array([E(img) for img in images], dtype=float)
        grads = np.array([np.asarray(gradE(img), dtype=float) for img in images], dtype=float)

        taus = _compute_tangents(images, energies, tangent=tangent)

        d_plus = images[2:] - images[1:-1]
        d_minus = images[1:-1] - images[:-2]
        l_plus = np.linalg.norm(d_plus, axis=1)
        l_minus = np.linalg.norm(d_minus, axis=1)

        ci: int | None = None
        if climb and it >= climb_start:
            ci = int(1 + np.argmax(energies[1:-1]))

        max_force = 0.0
        for i in range(1, m_count - 1):
            tau = taus[i]
            g = grads[i]
            g_par = np.dot(g, tau) * tau
            g_perp = g - g_par
            f_spring = k_spring * (l_plus[i - 1] - l_minus[i - 1]) * tau
            f_neb = -g_perp + f_spring

            if ci is not None and i == ci:
                f_neb = -g + 2.0 * g_par

            max_force = max(max_force, _norm(f_neb))
            images[i] = images[i] + step_size * f_neb

        if reparam_every > 0 and (it + 1) % reparam_every == 0:
            images = redistribute(images)

        if verbose and (it % 50 == 0 or it == max_steps - 1):
            e0 = min(float(energies[0]), float(energies[-1]))
            barrier = float(np.max(energies) - e0)
            print(f"it={it:5d}  max|F|={max_force:.3e}  barrier~{barrier:.6g}")

        if max_force < force_tol:
            break

        if max_force > 10.0 * last_max_force and step_size > 1e-6:
            step_size *= 0.5
        last_max_force = max_force

    energies = np.array([E(img) for img in images], dtype=float)
    e0 = min(float(energies[0]), float(energies[-1]))
    barrier = float(np.max(energies) - e0)
    saddle_index = int(np.argmax(energies))
    return {
        "images": images,
        "energies": energies,
        "barrier": barrier,
        "saddle_index": saddle_index,
        "iterations": iterations,
    }