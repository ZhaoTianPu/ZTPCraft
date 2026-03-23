from __future__ import annotations

def hermite_complex(n: int, z: complex) -> complex: ...
def cprefactor(n_i: int, n_j: int, phi_0_i: float, phi_0_j: float) -> float: ...
def cSij(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
) -> float: ...
def ccosij(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
    a: float,
    phi_ext: float,
) -> float: ...
def cn2ij(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
) -> float: ...
def cn2ij_GH(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
) -> float: ...
def cphi2ij(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
) -> float: ...
def cSij_GH(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
) -> float: ...
def cphi2ij_GH(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
) -> float: ...
def ccosij_plain_GH(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
    a: float,
    phi_ext: float,
) -> float: ...
def ccosij_complex_GH(
    n_i: int,
    n_j: int,
    phi_ratio_i: float,
    phi_ratio_j: float,
    phi_0_i: float,
    phi_0_j: float,
    a: float,
    phi_ext: float,
) -> float: ...
