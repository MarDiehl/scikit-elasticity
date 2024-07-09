# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
from scipy import special

def _xi(a,N_zeta3,N_omega):
    # https://doi.org/10.1007/BF00370053
    # eq (3)
    zeta3s, ws_zeta3 = special.roots_legendre(N_zeta3)
    omegas, ws_omega = (_*np.pi for _ in special.roots_legendre(N_omega))
    omegas+=np.pi
    p = np.meshgrid(zeta3s,omegas,indexing='ij')
    w = np.stack(np.meshgrid(ws_zeta3,ws_omega,indexing='ij'),axis=-1)
    xi = np.stack([(1-p[0]**2)**.5*np.cos(p[1])/a[0],
                   (1-p[0]**2)**.5*np.sin(p[1])/a[1],
                   p[0]/a[2]],axis=-1)
    return xi, np.prod(w,axis=-1)

def _G(C,xi):
    """
    Calculate the G tensor.

    Parameters
    ----------
    C : numpy.ndarray, shape (3,3,3,3)
        Stiffness
    xi : np.ndarray of shape (:,:,3)
    # https://doi.org/10.1007/BF00370053
    # eq (3)
    """
    e = np.zeros((3, 3, 3))
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = +1.0                                                     # Levi-Civita symbol
    e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1.0

    K = np.einsum('ijkl,...j,...l',C,xi,xi)
    N = 0.5*np.einsum('ikl,jmn,...km,...ln',e,e,K,K)
    D = np.sum(N[...,0:1,:]*K[...,0:1,:],axis=-1,keepdims=True)

    return np.einsum('...k,...l,...ij',xi,xi,N/D)


def eshelby(C,*,
            a=None,N_zeta3=None,N_omega=None,
            xi=None,w=None):
    """
    Calculate the Eshelby tensor numerically.

    Parameters
    ----------
    C : numpy.ndarray, shape (3,3,3,3)
        Stiffness
    a : sequence of int, len (3), optional
        Axis length
    N_zeta3 : integer, optional
        Number of integration points
    N_omega : integer, optional
        Number of integration points
    xi : np.ndarray of shape (:,:,3)
    w : np.ndarray of shape (:,:,3)
        Gaussian weights

    Notes
    -----
    This function can be called with either the axis and
    number of integration points or with precalculated.

    References
    ----------
    Gavazzi, A.C., Lagoudas, D.C.
    On the numerical evaluation of Eshelby's tensor and its application to elastoplastic fibrous composites.
    Computational Mechanics 7, 13–19 (1990).
    https://doi.org/10.1007/BF00370053
    """
    if xi is None and w is None:
        xi, w = _xi(a,N_zeta3,N_omega)
    elif N_zeta3 is None and N_omega is None and a is None:
        pass
    else:
        raise ValueError
    G = _G(C,xi)
    return np.sum(np.einsum('mnkl,...imjn,...',C,G,w)+np.einsum('mnkl,...jmin,...',C,G,w),axis=(0,1))/(8*np.pi) # eq. (4)
