"""
Cython-accelerated oscillator integral routines used by Fluxonium non-orthogonal basis calculations.
This module provides:
- hermite_complex
- cprefactor
- cSij, cn2ij, cphi2ij, ccosij (analytical forms)
- cSij_GH, cn2ij_GH, cphi2ij_GH, ccosij_complex_GH (Gauss-Hermite quadrature forms)
"""
# imports
from libc.math cimport floor, sqrt, pi, exp, cos, abs
cdef extern from "complex.h":
    double complex cexp(double complex)
    double complex csqrt(double complex)
    double creal(double complex)
from cython.view cimport array as cvarray

from cython cimport Py_ssize_t

from scipy.special.cython_special cimport eval_hermite, gamma, hyp1f1

import numpy as np
from scipy.special import roots_hermite


# generate Gauss-Hermite quadrature points and weights
GHdata = np.zeros((2,300,300))
for index in range(1,300):
    zeros, weights = roots_hermite(index)
    zeros = np.pad(zeros, (0,300-index))
    weights = np.pad(weights, (0,300-index))
    GHdata[0][index][:] = zeros
    GHdata[1][index][:] = weights

# create a memoryview object to access GH quadrature data
cdef double [:, :, :] GHdata_view = GHdata

# continuous product function
cdef unsigned long int cprod(Py_ssize_t nstart, Py_ssize_t nfinal, Py_ssize_t dn):
    cdef unsigned long int prod_result
    cdef Py_ssize_t n = nstart + dn
    if nstart <= 0:
        prod_result = 1
    else:
        prod_result = nstart
    if nfinal == 0:
        while n > nfinal:
            prod_result *= n
            n += dn
    else:
        while n >= nfinal:
            prod_result *= n
            n += dn
    return prod_result

# Hermite polynomial with complex argument, using hyp1f1 functions
# notice that there are known numerical issues with this function when n is large
cpdef double complex hermite_complex(Py_ssize_t n, double complex z):
    cdef Py_ssize_t n_2 = n//2
    cdef double sign = 1 - 2* (n_2%2)
    cdef double prefactor = gamma(n + 1.)/gamma(n_2 + 1.)
    cdef double complex z2 = z*z
    if n%2 == 0:
        return sign * prefactor * hyp1f1(-n_2, 0.5, z2)
    else:
        return sign * prefactor * 2 * z * hyp1f1(-n_2, 1.5, z2)

# combination function
cdef unsigned long int ccomb(Py_ssize_t n, Py_ssize_t r):
    if n==r:
        return 1
    elif r==0:
        return 1
    else:
        return cprod(n,n-r+1,-1)//cprod(r,0,-1)

# the result of the Gaussian integral without the exponential part
cdef double cGauss_int_without_exp(Py_ssize_t n, double a, double b):
    cdef double Gauss_sum
    cdef Py_ssize_t k
    Gauss_sum = 0
    for k in range(round(floor(n/2.+1))):
        Gauss_sum += ccomb(n,2*k)*cprod(2*k-1, 0, -2)/(2*a)**(n-k)*b**(n-2*k)
    Gauss_sum *= sqrt(pi/a)
    return Gauss_sum

# the result of the Gaussian integral without the exponential part, for complex coefficients
cdef long double complex cGauss_int_without_exp_complex(Py_ssize_t n, double complex a, double complex b):
    cdef long double complex Gauss_sum
    cdef Py_ssize_t k
    Gauss_sum = 0
    for k in range(round(floor(n/2.+1))):
        Gauss_sum += ccomb(n,2*k)*cprod(2*k-1, 0, -2)/(2*a)**(n-k)*b**(n-2*k)
    Gauss_sum *= csqrt(pi/a)
    return Gauss_sum

# prefactor of the matrix element integrals from the two harmonic oscillator states
cpdef double cprefactor(Py_ssize_t n_i,Py_ssize_t n_j, double phi_0_i, double phi_0_j):
    return (1./sqrt(2.**n_i)) * 1./sqrt(gamma(<double>n_i + 1.) )*\
           (1./sqrt(2.**n_j)) * 1./sqrt(gamma(<double>n_j + 1.) )*\
           (1./sqrt(pi * phi_0_i * phi_0_j))

# matrix elements of the overlapping integral, using the analytical form
cpdef double cSij(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j):
    cdef double Sij_sum_j = 0
    cdef double Sij_sum_ij = 0
    cdef double expfactor = -phi_ratio_i**2/2 -phi_ratio_j**2/2 + (phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)*(phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)/(4*(1/(2*phi_0_i**2)+1/(2*phi_0_j**2)))
    cdef double expvalue = exp(expfactor)
    cdef Py_ssize_t k_i
    cdef Py_ssize_t k_j
    if (phi_ratio_i==phi_ratio_j)&(phi_0_i==phi_0_j):
        if (n_i==n_j):
            return 1
        else:
            return 0
    else:
        for k_i in range(n_i+1):
            Sij_sum_j = 0
            for k_j in range(n_j+1):
                Sij_sum_j +=\
                ccomb(n_j,k_j)*eval_hermite(n_j-k_j,-phi_ratio_j)\
                *(2/phi_0_j)**k_j\
                *cGauss_int_without_exp(k_i+k_j,1/(2*phi_0_i**2)+1/(2*phi_0_j**2),phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)
            Sij_sum_ij += Sij_sum_j*ccomb(n_i,k_i)*eval_hermite(n_i-k_i,-phi_ratio_i)*(2/phi_0_i)**k_i
        return Sij_sum_ij*cprefactor(n_i,n_j,phi_0_i,phi_0_j)*expvalue

# matrix elements of the cos(a*phi - phi_ext), using the analytical form
cpdef double ccosij(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j, double a, double phi_ext):
    cdef double complex cosij_sum_j = 0
    cdef double complex cosij_sum_ij = 0
    cdef double complex expfactor = -1j*phi_ext -phi_ratio_i**2/2 -phi_ratio_j**2/2 + (1j*a + phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)*(1j*a + phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)/(4*(1/(2*phi_0_i**2)+1/(2*phi_0_j**2)))
    cdef double complex expvalue = cexp(expfactor)
    cdef Py_ssize_t k_i
    cdef Py_ssize_t k_j
    for k_i in range(n_i+1):
        cosij_sum_j = 0
        for k_j in range(n_j+1):
            cosij_sum_j +=\
            ccomb(n_j,k_j)*eval_hermite(n_j-k_j,-phi_ratio_j)\
            *(2/phi_0_j)**k_j\
            *cGauss_int_without_exp_complex(k_i+k_j,1/(2*phi_0_i**2)+1/(2*phi_0_j**2),1j*a+phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)
        cosij_sum_ij += cosij_sum_j*ccomb(n_i,k_i)*eval_hermite(n_i-k_i,-phi_ratio_i)*(2/phi_0_i)**k_i
    return creal(cosij_sum_ij*cprefactor(n_i,n_j,phi_0_i,phi_0_j)*expvalue)

# matrix elements of n^2, using the analytical form
cpdef double cn2ij(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j):
    if (phi_ratio_i==phi_ratio_j)&(phi_0_i==phi_0_j):
        if (n_i==n_j):
            return -1/(2*phi_0_j**2)*(2*n_j+1)
        elif (n_i==n_j+2):
            return  1/(2*phi_0_j**2)*sqrt((n_j+1)*(n_j+2))
        elif (n_i==n_j-2):
            return  1/(2*phi_0_j**2)*sqrt(n_j*(n_j-1))
        else:
            return 0
    else:
        if n_j>1:
            return 1/(2*phi_0_j**2)*\
            (\
             sqrt(n_j    *(n_j-1))*cSij(n_i,n_j-2,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)\
            +sqrt((n_j+1)*(n_j+2))*cSij(n_i,n_j+2,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)\
            -    (2*n_j+1        )*cSij(n_i,n_j,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)
            )
        else:
            return 1/(2*phi_0_j**2)*\
            (\
             sqrt((n_j+1)*(n_j+2))*cSij(n_i,n_j+2,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)\
            -    (2*n_j+1        )*cSij(n_i,n_j,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)
            )

cpdef double cn2ij_GH(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j):
    if (phi_ratio_i==phi_ratio_j)&(phi_0_i==phi_0_j):
        if (n_i==n_j):
            return -1/(2*phi_0_j**2)*(2*n_j+1)
        elif (n_i==n_j+2):
            return  1/(2*phi_0_j**2)*sqrt((n_j+1.)*(n_j+2.))
        elif (n_i==n_j-2):
            return  1/(2*phi_0_j**2)*sqrt(n_j*(n_j-1.))
        else:
            return 0
    else:
        if n_j>1:
            return 1/(2*phi_0_j**2)*\
            (\
             sqrt(n_j    *(n_j-1))*cSij_GH(n_i,n_j-2,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)\
            +sqrt((n_j+1)*(n_j+2))*cSij_GH(n_i,n_j+2,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)\
            -    (2*n_j+1        )*cSij_GH(n_i,n_j,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)
            )
        else:
            return 1/(2*phi_0_j**2)*\
            (\
             sqrt((n_j+1)*(n_j+2))*cSij_GH(n_i,n_j+2,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)\
            -    (2*n_j+1        )*cSij_GH(n_i,n_j,phi_ratio_i,phi_ratio_j,phi_0_i,phi_0_j)
            )

# matrix elements of phi^2, using the analytical form
cpdef double cphi2ij(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j):
    cdef double phi2ij_sum_j = 0
    cdef double phi2ij_sum_ij = 0
    cdef double expfactor = -phi_ratio_i**2/2 -phi_ratio_j**2/2 + (phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)*(phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)/(4*(1/(2*phi_0_i**2)+1/(2*phi_0_j**2)))
    cdef double expvalue = exp(expfactor)
    if (phi_ratio_i==phi_ratio_j)&(phi_ratio_i==0)&(phi_0_i==phi_0_j)&(n_i!=n_j)&(n_i!=n_j+2)&(n_i!=n_j-2):
        return 0
    else:
        for k_i in range(n_i+1):
            phi2ij_sum_j = 0
            for k_j in range(n_j+1):
                phi2ij_sum_j +=\
                ccomb(n_j,k_j)*eval_hermite(n_j-k_j,-phi_ratio_j)\
                *(2/phi_0_j)**k_j\
                *cGauss_int_without_exp(k_i+k_j+2,1/(2*phi_0_i**2)+1/(2*phi_0_j**2),phi_ratio_i/phi_0_i+phi_ratio_j/phi_0_j)
            phi2ij_sum_ij += phi2ij_sum_j*ccomb(n_i,k_i)*eval_hermite(n_i-k_i,-phi_ratio_i)*(2/phi_0_i)**k_i
    return phi2ij_sum_ij*cprefactor(n_i,n_j,phi_0_i,phi_0_j)*expvalue

# evaluating the overlap matrix elements with Gauss-Hermite quadrature
cpdef double cSij_GH(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j):
    """ 
    Evaluating the overlap matrix elements with Gauss-Hermite quadrature

    PARAMETERS
    ----------
        n_i : int
            the excitation number of the state i
        n_j : int
            the excitation number of the state j
        phi_ratio_i : double
            the ratio of the center displacement of the state i to the width (phi_0_i) of the state i
        phi_ratio_j : double
            the ratio of the center displacement of the state j to the width (phi_0_j) of the state j
        phi_0_i : double
            the width of the state i
        phi_0_j : double
            the width of the state j
    
    RETURNS
    -------
        the overlap matrix element of the state i and j
    """
    # initialization
    cdef double A = 0
    cdef double Asqrt = 0
    cdef double B = 0
    cdef double B2 = 0
    cdef double B_2A = 0
    cdef double exparg = 0
    cdef Py_ssize_t order = 0
    cdef double GHintegral = 0
    cdef double phi_temp = 0
    # calculate some useful numbers
    A = 1./(2.*phi_0_i*phi_0_i) + 1./(2.*phi_0_j*phi_0_j)
    Asqrt = sqrt(A)
    B = phi_ratio_i / phi_0_i + phi_ratio_j / phi_0_j
    B2 = B*B 
    B_2A = B/(2. * A)
    exparg = B2/(4. * A) - phi_ratio_i * phi_ratio_i / 2. - phi_ratio_j * phi_ratio_j / 2. 
    # compute the integral with Gauss-Hermite quadrature
    # ceil((n_i + n_j + 1)/2)
    order = int(-(-((n_i + n_j +1)/2.)//1)) 
    for term in range(order):
        phi_temp = GHdata_view[0,order,term]/Asqrt + B_2A
        GHintegral += GHdata_view[1,order,term] *\
            eval_hermite(n_i, phi_temp/phi_0_i - phi_ratio_i) *\
            eval_hermite(n_j, phi_temp/phi_0_j - phi_ratio_j) 
    return exp(exparg) * cprefactor(n_i,n_j,phi_0_i,phi_0_j)/Asqrt * GHintegral

# evaluating the phi^2 matrix elements with Gauss-Hermite quadrature
cpdef double cphi2ij_GH(Py_ssize_t n_i,Py_ssize_t n_j,double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j):
    """ 
    Evaluating the phi^2 matrix elements with Gauss-Hermite quadrature

    PARAMETERS
    ----------
        n_i : int
            the excitation number of the state i
        n_j : int
            the excitation number of the state j
        phi_ratio_i : double
            the ratio of the center displacement of the state i to the width (phi_0_i) of the state i
        phi_ratio_j : double
            the ratio of the center displacement of the state j to the width (phi_0_j) of the state j
        phi_0_i : double
            the width of the state i
        phi_0_j : double
            the width of the state j
    
    RETURNS
    -------
        the phi^2 matrix element of the state i and j
    """
    cdef double A = 0
    cdef double Asqrt = 0
    cdef double B = 0
    cdef double B2 = 0
    cdef double B_2A = 0
    cdef double exparg = 0
    cdef Py_ssize_t order = 0
    cdef double GHintegral = 0
    cdef double phi_temp = 0
    # calculate some useful numbers
    A = 1./(2.*phi_0_i*phi_0_i) + 1./(2.*phi_0_j*phi_0_j)
    Asqrt = sqrt(A)
    B = phi_ratio_i / phi_0_i + phi_ratio_j / phi_0_j
    B2 = B*B 
    B_2A = B/(2. * A)
    exparg = B2/(4. * A) - phi_ratio_i * phi_ratio_i / 2. - phi_ratio_j * phi_ratio_j / 2. 
    # Gauss-Hermite quadrature
    order = int(-(-((n_i + n_j +1)/2.)//1)) + 1
    for term in range(order):
        phi_temp = GHdata_view[0,order,term]/Asqrt + B_2A
        GHintegral += GHdata_view[1,order,term] *\
            eval_hermite(n_i, phi_temp/phi_0_i - phi_ratio_i) *\
            eval_hermite(n_j, phi_temp/phi_0_j - phi_ratio_j) *\
            phi_temp*phi_temp # this is the phi^2 operator
    return exp(exparg) * cprefactor(n_i,n_j,phi_0_i,phi_0_j)/Asqrt * GHintegral

# evaluating the cosine (cos(a*phi - phi_ext)) matrix elements with Gauss-Hermite quadrature, here we use the plain cosine
cpdef double ccosij_plain_GH(Py_ssize_t n_i, Py_ssize_t n_j, double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j, a, phi_ext):
    """ 
    Evaluating the the cosine (cos(a*phi - phi_ext)) matrix elements with Gauss-Hermite quadrature, here we use the plain cosine

    PARAMETERS
    ----------
        n_i : int
            the excitation number of the state i
        n_j : int
            the excitation number of the state j
        phi_ratio_i : double
            the ratio of the center displacement of the state i to the width (phi_0_i) of the state i
        phi_ratio_j : double
            the ratio of the center displacement of the state j to the width (phi_0_j) of the state j
        phi_0_i : double
            the width of the state i
        phi_0_j : double
            the width of the state j
        a : double
            the coefficient of phi in the cosine operator
        phi_ext : double
            the external flux
    
    RETURNS
    -------
        the cos(a*phi - phi_ext) matrix element of the state i and j
    """
    cdef double A = 0
    cdef double Asqrt = 0
    cdef double B = 0
    cdef double B2 = 0
    cdef double B_2A = 0
    cdef double exparg = 0
    cdef double GHintegral = 0
    cdef double phi_temp = 0
    cdef double cosine_order = 0
    # calculate some useful numbers
    A = 1./(2.*phi_0_i*phi_0_i) + 1./(2.*phi_0_j*phi_0_j)
    Asqrt = sqrt(A)
    B = phi_ratio_i / phi_0_i + phi_ratio_j / phi_0_j
    B2 = B*B 
    B_2A = B/(2. * A)
    exparg = B2/(4. * A) - phi_ratio_i * phi_ratio_i / 2. - phi_ratio_j * phi_ratio_j / 2. 
    # Gauss-Hermite quadrature
    # to capture the cosine term better, add an extra 6*sigma/(period/2)/2 number of orders
    cosine_order = (n_i*phi_0_i+n_j*phi_0_j + abs(phi_ratio_i*phi_0_i - phi_ratio_j*phi_0_j)) * 2 * a /pi
    order = int(-(-((n_i + n_j + 1 + 2*cosine_order )/2.)//1)) 
    for term in range(order):
        phi_temp = GHdata_view[0,order,term]/Asqrt + B_2A
        GHintegral += GHdata_view[1,order,term] *\
            eval_hermite(n_i, phi_temp/phi_0_i - phi_ratio_i) *\
            eval_hermite(n_j, phi_temp/phi_0_j - phi_ratio_j) *\
            cos(a* phi_temp - phi_ext)
    return exp(exparg) * cprefactor(n_i,n_j,phi_0_i,phi_0_j)/Asqrt * GHintegral

# evaluating the cosine (cos(a*phi - phi_ext)) matrix elements with Gauss-Hermite quadrature by writing the cosine in terms of complex exponentials
# then group the cosine with the exponentials; this way the polynomial become complex
cpdef double ccosij_complex_GH(Py_ssize_t n_i, Py_ssize_t n_j, double phi_ratio_i,double phi_ratio_j,double phi_0_i,double phi_0_j, double a, double phi_ext):
    """ 
    Evaluating the the cosine (cos(a*phi - phi_ext)) matrix elements with Gauss-Hermite quadrature by writing the cosine in terms of complex exponentials
    then group the cosine with the exponentials; this way the polynomial become complex

    PARAMETERS
    ----------
        n_i : int
            the excitation number of the state i
        n_j : int
            the excitation number of the state j
        phi_ratio_i : double
            the ratio of the center displacement of the state i to the width (phi_0_i) of the state i
        phi_ratio_j : double
            the ratio of the center displacement of the state j to the width (phi_0_j) of the state j
        phi_0_i : double
            the width of the state i
        phi_0_j : double
            the width of the state j
        a : double
            the coefficient of phi in the cosine operator
        phi_ext : double
            the external flux
    
    RETURNS
    -------
        the cos(a*phi - phi_ext) matrix element of the state i and j
    """
    cdef double A = 0
    cdef double Asqrt = 0
    cdef double complex B = 0
    cdef double complex B2 = 0
    cdef double complex B_2A = 0
    cdef double complex exparg = 0
    cdef double complex GHintegral = 0
    cdef double complex phi_temp = 0
    cdef Py_ssize_t order = 0
    # calculate some useful numbers
    A = 1./(2.*phi_0_i*phi_0_i) + 1./(2.*phi_0_j*phi_0_j)
    Asqrt = sqrt(A)
    B = phi_ratio_i / phi_0_i + phi_ratio_j / phi_0_j + 1j*a
    B2 = B*B 
    B_2A = B/(2. * A)
    exparg = B2/(4. * A) - phi_ratio_i * phi_ratio_i / 2. - phi_ratio_j * phi_ratio_j / 2. 
    # Gauss-Hermite quadrature
    order = int(-(-((n_i + n_j +1)/2.)//1)) + 1
    for term in range(order):
        phi_temp = GHdata_view[0,order,term]/Asqrt + B_2A
        GHintegral += GHdata_view[1,order,term] *\
            hermite_complex(n_i, phi_temp/phi_0_i - phi_ratio_i) *\
            hermite_complex(n_j, phi_temp/phi_0_j - phi_ratio_j) 
    return creal(cexp(exparg - 1j*phi_ext) * cprefactor(n_i,n_j,phi_0_i,phi_0_j)/Asqrt * GHintegral)