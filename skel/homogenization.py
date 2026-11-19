# SPDX-License-Identifier: AGPL-3.0-or-later
r"""
Homogenization of elastic stiffness for ensembles of orientations.

The functions in this module take a set of orientations representing
the texture together with either stiffness :math:`C` or compliance :math:`S`
of a single crystal and calculate the effective stiffness or compliance
in Mandel notation.

The orientations ``r`` must be a one-dimensional ``damask.Rotation``.

The following schemes are available:

- Voigt (isostrain) and Reuss (isostress) bounds
- Hill averages, including arithmetic and bound-based geometric averages
- superHill approximation
- Orientation-wise geometric mean
- Self-consistent scheme
"""

import numpy as np
from scipy import linalg, optimize

from . import Mandel
from . import Eshelby


# Primitives on rotated data
def _C_Voigt(C_rotated):
    return np.average(C_rotated,axis=0)

def _S_Voigt(C_rotated):
    return np.linalg.inv(_C_Voigt(C_rotated))

def _S_Reuss(S_rotated):
    return np.average(S_rotated,axis=0)

def _C_Reuss(S_rotated):
    return np.linalg.inv(_S_Reuss(S_rotated))

def _C_Hill_C(C_rotated,S_rotated):
    return (_C_Voigt(C_rotated)+_C_Reuss(S_rotated))*.5

def _S_Hill_S(C_rotated,S_rotated):
    return (_S_Voigt(C_rotated)+_S_Reuss(S_rotated))*.5

def _C_geometric_log(C_rotated,S_rotated):
    return linalg.expm((linalg.logm(_C_Voigt(C_rotated))+linalg.logm(_C_Reuss(S_rotated)))*.5)

def _C_geometric_affine(C_rotated,S_rotated):
    C_Voigt = _C_Voigt(C_rotated)
    C_Reuss = _C_Reuss(S_rotated)
    C_sqrt = linalg.sqrtm(C_Voigt)
    C_inv_sqrt = np.linalg.inv(C_sqrt)
    return C_sqrt@linalg.sqrtm(C_inv_sqrt@C_Reuss@C_inv_sqrt)@C_sqrt

def _C_super_hill(C_rotated,S_rotated,tol,maxiter):
    C = _C_Hill_C(C_rotated,S_rotated)
    S = _S_Hill_S(C_rotated,S_rotated)
    for _ in range(maxiter):
        C_next = (C+np.linalg.inv(S))*.5
        S_next = (S+np.linalg.inv(C))*.5
        if (np.linalg.norm(C_next-C) <= tol*np.linalg.norm(C_next)
            and np.linalg.norm(S_next-S) <= tol*np.linalg.norm(S_next)):
            return C_next
        C, S = C_next, S_next
    raise RuntimeError('super_hill did not converge')

def _C_geo(C_rotated):
    try:
        lnC = linalg.logm(C_rotated)
    except ValueError:
        # Older SciPy versions do not support batched matrix logarithms.
        eigenvalues, eigenvectors = np.linalg.eigh(C_rotated)
        lnC = (eigenvectors*np.log(eigenvalues)[...,None,:])@np.swapaxes(eigenvectors,-1,-2)
    return linalg.expm(np.average(lnC,axis=0))


def C_Voigt(r,*,C=None,S=None):
    """
    Homogenized stiffness according to Voigt (isostrain).

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    C_Voigt : numpy.ndarray of shape (6,6)
        Homogenized stiffness in Mandel notation.

    References
    ----------
    W. Voigt, Annalen der Physik und Chemie 274(12):573-587
    https://doi.org/10.1002/andp.18892741201
    """
    if (C is None) == (S is None):
        raise KeyError('provide exactly one of "C" or "S"')
    if C is None:
        C = np.linalg.inv(S)
    return _C_Voigt(Mandel.rotate_stiffness(r,C))

def S_Voigt(r,*,C=None,S=None):
    """
    Homogenized compliance according to Voigt (isostrain).

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    S_Voigt : numpy.ndarray of shape (6,6)
        Homogenized compliance in Mandel notation.

    References
    ----------
    W. Voigt, Annalen der Physik und Chemie 274(12):573-587
    https://doi.org/10.1002/andp.18892741201
    """
    return np.linalg.inv(C_Voigt(r,C=C,S=S))


def S_Reuss(r,*,C=None,S=None):
    """
    Homogenized compliance according to Reuss (isostress).

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    S_Reuss : numpy.ndarray of shape (6,6)
        Homogenized compliance in Mandel notation.

    References
    ----------
    A. Reuss, ZAMM 9(1):49-58
    https://doi.org/10.1002/zamm.19290090104
    """
    if (C is None) == (S is None):
        raise KeyError('provide exactly one of "C" or "S"')
    if S is None:
        S = np.linalg.inv(C)
    return _S_Reuss(Mandel.rotate_compliance(r,S))

def C_Reuss(r,*,C=None,S=None):
    """
    Homogenized stiffness according to Reuss (isostress).

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    C_Reuss : numpy.ndarray of shape (6,6)
        Homogenized stiffness in Mandel notation.

    References
    ----------
    A. Reuss, ZAMM 9(1):49-58
    https://doi.org/10.1002/zamm.19290090104
    """
    return np.linalg.inv(S_Reuss(r,C=C,S=S))


def C_Hill(r,*,C=None,S=None,method='super_hill',tol=1e-10,maxiter=100):
    """
    Homogenized stiffness according to a Hill-type scheme.

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.
    method : {'arithmetic_stiffness', 'arithmetic_compliance',
              'geometric_log', 'geometric_affine', 'super_hill'}, optional
        Hill-type estimate to return. Defaults to ``super_hill``.
    tol : float, optional
        Relative convergence tolerance for ``super_hill``.
        Defaults to 1e-10.
    maxiter : int, optional
        Maximum number of iterations for ``super_hill``. Defaults to 100.

    Returns
    -------
    C_Hill : numpy.ndarray of shape (6,6)
        Homogenized stiffness in Mandel notation.

    Notes
    -----
    ``arithmetic_stiffness`` uses the average of C from Voigt and Reuss,
    ``arithmetic_compliance`` uses the inverse of the average of S,
    ``geometric_log`` and ``geometric_affine`` use the corresponding
    geometric mean of the Voigt and Reuss stiffness estimates, and
    ``super_hill`` uses the iterative superHill approximation. The
    orientation-wise geometric mean is available as :func:`C_geo`.

    References
    ----------
    R. Hill, Proceedings of the Physical Society. Section A 65(5):349-354, 1952
    https://doi.org/10.1088/0370-1298/65/5/307

    K. S. Aleksandrov and L. A. Aizenberg, Doklady Akademii Nauk SSSR 167(5):1028-1031, 1966 (in Russian)
    https://geodesic.mathdoc.fr/item/DAN_1966_167_5_a17/

    W. Pusz and S. L. Woronowicz, Reports on Mathematical Physics 8(2):159-170, 1975
    https://doi.org/10.1016/0034-4877(75)90061-0

    F. Kubo and T. Ando, Mathematische Annalen 246(3):205-224, 1980
    https://doi.org/10.1007/BF01371042

    S. Matthies and M. Humbert, Journal of Applied Crystallography 28(3):254-266, 1995
    https://doi.org/10.1107/S0021889894009623

    V. Arsigny et al., Magnetic Resonance in Medicine 56(2):411-421, 2006
    https://doi.org/10.1002/mrm.20965
    """
    methods = ('arithmetic_stiffness','arithmetic_compliance',
               'geometric_log','geometric_affine','super_hill')
    if method not in methods:
        raise KeyError(f'method must be one of {", ".join(methods)}')
    if (C is None) == (S is None):
        raise KeyError('provide exactly one of "C" or "S"')
    if C is None:
        C = np.linalg.inv(S)
    if S is None:
        S = np.linalg.inv(C)
    C_rotated = Mandel.rotate_stiffness(r,C)
    S_rotated = Mandel.rotate_compliance(r,S)
    if method == 'arithmetic_stiffness':
        return _C_Hill_C(C_rotated,S_rotated)
    if method == 'arithmetic_compliance':
        return np.linalg.inv(_S_Hill_S(C_rotated,S_rotated))
    if method == 'geometric_log':
        return _C_geometric_log(C_rotated,S_rotated)
    if method == 'geometric_affine':
        return _C_geometric_affine(C_rotated,S_rotated)
    return _C_super_hill(C_rotated,S_rotated,tol,maxiter)

def S_Hill(r,*,C=None,S=None,method='super_hill',tol=1e-10,maxiter=100):
    """
    Homogenized compliance according to a Hill-type scheme.

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.
    method : {'arithmetic_stiffness', 'arithmetic_compliance',
              'geometric_log', 'geometric_affine', 'super_hill'}, optional
        Hill-type estimate to return. Defaults to ``super_hill``.
    tol : float, optional
        Relative convergence tolerance for ``super_hill``.
        Defaults to 1e-10.
    maxiter : int, optional
        Maximum number of iterations for ``super_hill``. Defaults to 100.

    Returns
    -------
    S_Hill : numpy.ndarray of shape (6,6)
        Homogenized compliance in Mandel notation.

    Notes
    -----
    ``arithmetic_stiffness`` uses the inverse of the average of C,
    ``arithmetic_compliance`` uses the average of S,
    ``geometric_log`` and ``geometric_affine`` use the corresponding
    geometric mean of the Voigt and Reuss compliance estimates, and
    ``super_hill`` uses the inverse of the iterative superHill estimate.
    The orientation-wise geometric mean is available as :func:`S_geo`.

    This function returns the inverse of :func:`C_Hill` for the selected
    method.

    References
    ----------
    R. Hill, Proceedings of the Physical Society. Section A 65(5):349-354, 1952
    https://doi.org/10.1088/0370-1298/65/5/307

    K. S. Aleksandrov and L. A. Aizenberg, Doklady Akademii Nauk SSSR 167(5):1028-1031, 1966 (in Russian)
    https://geodesic.mathdoc.fr/item/DAN_1966_167_5_a17/

    W. Pusz and S. L. Woronowicz, Reports on Mathematical Physics 8(2):159-170, 1975
    https://doi.org/10.1016/0034-4877(75)90061-0

    F. Kubo and T. Ando, Mathematische Annalen 246(3):205-224, 1980
    https://doi.org/10.1007/BF01371042

    S. Matthies and M. Humbert, Journal of Applied Crystallography 28(3):254-266, 1995
    https://doi.org/10.1107/S0021889894009623

    V. Arsigny et al., Magnetic Resonance in Medicine 56(2):411-421, 2006
    https://doi.org/10.1002/mrm.20965
    """
    return np.linalg.inv(C_Hill(r,C=C,S=S,method=method,tol=tol,maxiter=maxiter))


def C_geo(r,*,C=None,S=None):
    r"""
    Homogenized stiffness according to the geometric mean.

    This is the orientation-wise geometric mean, also referred to as GEO.

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    C_geo : numpy.ndarray of shape (6,6)
        Homogenized stiffness in Mandel notation.

    Notes
    -----
    The geometric mean is calculated over all rotated stiffnesses,

    .. math::

        C_\mathrm{geo} = \exp\left(\langle \ln C\rangle\right),

    which satisfies :math:`C_\mathrm{geo}S_\mathrm{geo}=I`.

    References
    ----------
    K. S. Aleksandrov and L. A. Aizenberg, Doklady Akademii Nauk SSSR 167(5):1028-1031, 1966 (in Russian)
    https://geodesic.mathdoc.fr/item/DAN_1966_167_5_a17/

    S. Matthies and M. Humbert, Journal of Applied Crystallography 28(3):254-266, 1995
    https://doi.org/10.1107/S0021889894009623
    """
    if (C is None) == (S is None):
        raise KeyError('provide exactly one of "C" or "S"')
    if C is None:
        C = np.linalg.inv(S)
    return _C_geo(Mandel.rotate_stiffness(r,C))

def S_geo(r,*,C=None,S=None):
    """
    Homogenized compliance according to the geometric mean.

    This is the inverse of the orientation-wise geometric mean, also
    referred to as GEO.

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    S_geo : numpy.ndarray of shape (6,6)
        Homogenized compliance in Mandel notation.

    Notes
    -----
    This is the inverse of :func:`C_geo`.

    References
    ----------
    K. S. Aleksandrov and L. A. Aizenberg, Doklady Akademii Nauk SSSR 167(5):1028-1031, 1966 (in Russian)
    https://geodesic.mathdoc.fr/item/DAN_1966_167_5_a17/

    S. Matthies and M. Humbert, Journal of Applied Crystallography 28(3):254-266, 1995
    https://doi.org/10.1107/S0021889894009623
    """
    return np.linalg.inv(C_geo(r,C=C,S=S))


def C_sc(r,*,C=None,S=None):
    """
    Homogenized stiffness according to the self-consistent scheme.

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    C_sc : numpy.ndarray of shape (6,6)
        Homogenized stiffness in Mandel notation.

    References
    ----------
    https://doi.org/10.1002/zamm.201100135
    """
    if (C is None) == (S is None):
        raise KeyError('provide exactly one of "C" or "S"')
    if C is None:
        C = np.linalg.inv(S)
    if S is None:
        S = np.linalg.inv(C)
    C_rotated = Mandel.rotate_stiffness(r,C)
    S_rotated = Mandel.rotate_compliance(r,S)
    C_I = _C_Hill_C(C_rotated,S_rotated)

    # https://dx.doi.org/10.1002/zamm.201100135, eq. (4)
    def f(x,xis,w,C_II):
        E = Mandel.to_6x6(Eshelby.eshelby(Mandel.to_3x3x3x3(x),xi=xis,w=w))@np.linalg.inv(x)
        return np.average(C_II@np.linalg.inv(E@(C_II-x)+np.eye(6)),axis=0)

    xis, w = Eshelby._xi([1,1,1],8,64)
    return optimize.fixed_point(f,C_I,args=(xis,w,C_rotated),xtol=1e-4)

def S_sc(r,*,C=None,S=None):
    """
    Homogenized compliance according to the self-consistent scheme.

    Parameters
    ----------
    r : damask.Rotation of shape (n,)
        One-dimensional array of orientations.
    S|C : numpy.ndarray of shape (6,6)
        Stiffness or compliance matrix in Mandel notation.

    Returns
    -------
    S_sc : numpy.ndarray of shape (6,6)
        Homogenized compliance in Mandel notation.

    References
    ----------
    https://doi.org/10.1002/zamm.201100135
    """
    return np.linalg.inv(C_sc(r,C=C,S=S))
