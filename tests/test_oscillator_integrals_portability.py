"""Check the compiled kernel against independent numerical references."""
from math import factorial, pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermite

from ztpcraft.bosonic.oscillator_integrals import (
    _oscillator_integrals_1d_quadrature as kernel,
)


def wavefunction(n, ratio, width, x):
    u = x / width - ratio
    return eval_hermite(n, u) * np.exp(-u * u / 2) / sqrt(
        2**n * factorial(n) * sqrt(pi) * width
    )


@pytest.mark.parametrize("n", [0, 1, 2, 7, 14])
@pytest.mark.parametrize("z", [0.3 + 0.7j, -0.4 - 0.2j])
def test_complex_hermite_against_polynomial(n, z):
    coefficients = np.zeros(n + 1)
    coefficients[n] = 1
    expected = np.polynomial.hermite.hermval(z, coefficients)
    np.testing.assert_allclose(kernel.hermite_complex(n, z), expected, rtol=2e-12)


@pytest.mark.parametrize("levels", [(0, 0), (2, 3), (13, 2), (12, 13)])
def test_overlap_against_real_space_quadrature(levels):
    ni, nj = levels
    ri, rj, wi, wj = 0.25, -0.4, 0.9, 1.2
    expected, error = quad(
        lambda x: wavefunction(ni, ri, wi, x) * wavefunction(nj, rj, wj, x),
        -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11,
    )
    assert error < 1e-9
    np.testing.assert_allclose(
        kernel.cSij(ni, nj, ri, rj, wi, wj), expected, atol=2e-8, rtol=2e-7
    )


@pytest.mark.parametrize("levels", [(0, 0), (2, 3), (13, 2), (12, 13)])
@pytest.mark.parametrize("a,phase", [(0.7, 0.35), (-1.1, -0.6)])
def test_cosine_against_real_space_quadrature(levels, a, phase):
    ni, nj = levels
    ri, rj, wi, wj = 0.25, -0.4, 0.9, 1.2
    expected, error = quad(
        lambda x: wavefunction(ni, ri, wi, x) * wavefunction(nj, rj, wj, x)
        * np.cos(a * x - phase),
        -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11,
    )
    assert error < 1e-9
    for implementation in (kernel.ccosij, kernel.ccosij_complex_GH):
        np.testing.assert_allclose(
            implementation(ni, nj, ri, rj, wi, wj, a, phase),
            expected, atol=2e-8, rtol=2e-7,
        )
