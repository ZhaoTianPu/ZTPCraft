"""
Python API for oscillator integral routines and Fluxonium NOB assembly.
This module wraps the compiled Cython extension `_oscillator_integrals` and
exposes convenience functions used in notebooks.
"""

from __future__ import annotations

from math import factorial, pi, sqrt
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from . import _oscillator_integrals_1d_quadrature as _osc1d_ext  # type: ignore[reportMissingImports]

hermite_complex: Any = cast(Any, _osc1d_ext.hermite_complex)
cprefactor: Any = cast(Any, _osc1d_ext.cprefactor)
cSij: Any = cast(Any, _osc1d_ext.cSij)
ccosij: Any = cast(Any, _osc1d_ext.ccosij)
cn2ij: Any = cast(Any, _osc1d_ext.cn2ij)
cphi2ij: Any = cast(Any, _osc1d_ext.cphi2ij)
cSij_GH: Any = cast(Any, _osc1d_ext.cSij_GH)
cphi2ij_GH: Any = cast(Any, _osc1d_ext.cphi2ij_GH)
cn2ij_GH: Any = cast(Any, _osc1d_ext.cn2ij_GH)
ccosij_complex_GH: Any = cast(Any, _osc1d_ext.ccosij_complex_GH)

Array2D = npt.NDArray[np.float64]
Array1D = npt.NDArray[np.float64]

# Import Cython-accelerated functions
# try:
#     from ._oscillator_integrals import (
#         hermite_complex,
#         cprefactor,
#         cSij,
#         ccosij,
#         cn2ij,
#         cphi2ij,
#         cSij_GH,
#         cphi2ij_GH,
#         cn2ij_GH,
#         ccosij_complex_GH,
#     )
# except Exception as exc:  # pragma: no cover - useful diagnostic on import errors
#     raise RuntimeError(
#         "Failed to import ztpcraft.bosonic._oscillator_integrals. Make sure the Cython extension is built."
#     ) from exc


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


def lowdin(
    S: Array2D,
    H: Array2D,
    delta_cutoff: float,
) -> tuple[Array1D, Array2D]:
    eigvals, U = eigh(S)
    first_untruncated_basis: int = 0
    for index, delta_val in enumerate(eigvals):
        if delta_val > delta_cutoff:
            first_untruncated_basis = index
            break
    eigvals_arr: Array1D = np.asarray(eigvals, dtype=np.float64)
    U_arr: Array2D = np.asarray(U, dtype=np.float64)
    delta_truncated = eigvals_arr[first_untruncated_basis:]
    U_truncated = U_arr[:, first_untruncated_basis:]
    delta_inv_sqrt: Array2D = np.asarray(
        np.diag(1.0 / np.sqrt(delta_truncated)), dtype=np.float64
    )
    H_orth: Array2D = np.asarray(
        delta_inv_sqrt @ (U_truncated.T) @ H @ U_truncated @ delta_inv_sqrt,
        dtype=np.float64,
    )
    return eigvals_arr, H_orth


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
