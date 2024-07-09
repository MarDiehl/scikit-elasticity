# SPDX-License-Identifier: AGPL-3.0-or-later
import itertools

import numpy as np
from scipy import special

import skel

def eshelby_isotropic_sphere(nu):
    """Eshelby tensor for sphere in an isotropic medium."""
    # http://micro.stanford.edu/~caiwei/me340b/content/me340b-lecture02-v03.pdf
    # https://www.brown.edu/Departments/Engineering/Courses/EN224/eshelby/eshelby.html
    d = lambda i,j: 1 if i==j else 0
    S = np.zeros((3,3,3,3))
    for i,j,k,l in itertools.product(*[range(3)]*4):
        S[i,j,k,l] = (5*nu-1)/(15*(1-nu))*d(i,j)*d(k,l) \
                   + (4-5*nu)/(15*(1-nu))*(d(i,k)*d(j,l) + d(i,l)*d(j,k))
    return S


def test_isotropic_sphere():
    C_11 = 124.1e9
    C_12 = 93.7e9
    C66 = skel.Hooke.C('isotropic',C_11=C_11,C_12=C_12)
    C_44 = C66[3,3]
    nu = C_12*.5/(C_12+C_44)                                                                        # C_12 = λ, C_44 = G = μ
    C = skel.Voigt.to_3x3x3x3_stiffness(C66)
    S_iso = eshelby_isotropic_sphere(nu)
    assert np.allclose(S_iso,skel.Eshelby.eshelby(C,a=[1,1,1],N_zeta3=3,N_omega=100))


def test_xi_vectorization():
    n_zeta3 = n_omega = 24
    a = 1.1
    b = 1.5
    c = 3.2
    zeta3s, ws_zeta3 = special.roots_legendre(n_zeta3)
    omegas, ws_omega = (_*np.pi for _ in special.roots_legendre(n_omega))
    omegas+=np.pi

    xis, _ = skel.Eshelby._xi([a,b,c],n_zeta3,n_omega)
    for i,(zeta3,w_zeta3) in enumerate(zip(zeta3s,ws_zeta3)):
        for j,(omega,w_omega) in enumerate(zip(omegas,ws_omega)):
            xi = np.array([(1-zeta3**2)**.5*np.cos(omega)/a,
                           (1-zeta3**2)**.5*np.sin(omega)/b,
                           zeta3/c])
            assert np.allclose(xi,xis[i,j])

def test_G_vectorization():
    C_11 = 124.1e9
    C_12 = 93.7e9
    C_44 = 46.4e9
    C66 = skel.Hooke.C('cubic',C_11=C_11,C_12=C_12,C_44=C_44)
    C = skel.Voigt.to_3x3x3x3_stiffness(C66)
    n_zeta3 = n_omega = 40
    a = 1.1
    b = 1.4
    c = 3.0
    xis, w = skel.Eshelby._xi([a,b,c],n_zeta3,n_omega)
    Gs = skel.Eshelby._G(C,xis)
    for i in range(n_zeta3):
        for j in range(n_omega):
             assert np.allclose(skel.Eshelby._G(C,xis[i,j]),Gs[i,j])
