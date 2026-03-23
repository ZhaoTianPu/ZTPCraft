"""
Python API for oscillator integral routines and Fluxonium NOB assembly.
This module wraps the compiled Cython extension `_oscillator_integrals` and
exposes convenience functions used in notebooks.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import fsolve
from math import isclose, pi, sqrt, factorial
from copy import deepcopy

# Import Cython-accelerated functions
try:
    from ._oscillator_integrals import (
        hermite_complex,
        cprefactor,
        cSij,
        ccosij,
        cn2ij,
        cphi2ij,
        cSij_GH,
        cphi2ij_GH,
        cn2ij_GH,
        ccosij_complex_GH,
    )
except Exception as exc:  # pragma: no cover - useful diagnostic on import errors
    raise RuntimeError(
        "Failed to import ztpcraft.bosonic._oscillator_integrals. Make sure the Cython extension is built."
    ) from exc


# Oscillator energy and prefactors ---------------------------------------------


def OscEnergy(Vconst: float, V_d2phi_val: float, EC: float, n: int) -> float:
    return Vconst + (n + 1 / 2) * sqrt(8 * EC * V_d2phi_val)


def prefactor(n_i: int, n_j: int, phi_0_i: float, phi_0_j: float) -> float:
    return (
        1
        / np.sqrt(2**n_i * factorial(n_i) * np.sqrt(pi))
        * 1
        / np.sqrt(2**n_j * factorial(n_j) * np.sqrt(pi))
        * 1
        / np.sqrt(phi_0_i * phi_0_j)
    )


# Löwdin canonical orthogonalization -------------------------------------------


def lowdin(S: np.ndarray, H: np.ndarray, delta_cutoff: float):
    eigvals, U = eigh(S)
    first_untruncated_basis = 0
    for index, delta_val in enumerate(eigvals):
        if delta_val > delta_cutoff:
            first_untruncated_basis = index
            break
    delta_truncated = eigvals[first_untruncated_basis:]
    U_truncated = U[:, first_untruncated_basis:]
    delta_inv_sqrt = np.diag(1.0 / np.sqrt(delta_truncated))
    H_orth = delta_inv_sqrt @ (U_truncated.T) @ H @ U_truncated @ delta_inv_sqrt
    return eigvals, H_orth


__all__ = [
    # low-level Cython exports (via import *)
    "hermite_complex",
    "cprefactor",
    "cSij",
    "ccosij",
    "cn2ij",
    "cphi2ij",
    "cSij_GH",
    "cphi2ij_GH",
    "cn2ij_GH",
    "ccosij_complex_GH",
    # high-level helpers
    "OscEnergy",
    "prefactor",
    "lowdin",
]
