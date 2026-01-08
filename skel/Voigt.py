# SPDX-License-Identifier: AGPL-3.0-or-later
from . import _conventions
r"""
Convention according to W. Voigt for the compressed notation of symmetric tensors.

Notes
-----
The following order is used to store the unique elements of a
2nd order tensor (3x3 matrix) as a vector of length 6.

.. math::
    \vb{t}_\mathrm{Voigt} = (t_{xx}, t_{yy}, t_{zz}, t_{yz}, t_{xz}, t_{xy})


Different weights are used for the representation of stress and strain.
"""

def to_6_strain(epsilon):
    """
    Convert symmetric 2nd order strain tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    Returns
    -------
    epsilon_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Voigt notation.

    """
    return _conventions.convert_3x3_to_6(epsilon,2.)

def to_6_stress(sigma):
    """
    Convert symmetric 2nd order stress tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    Returns
    -------
    sigma_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Voigt notation.

    """
    return _conventions.convert_3x3_to_6(sigma,1.)


def to_6x6_compliance(S):
    """
    Convert symmetric 4th order compliance tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    Returns
    -------
    S_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in Voigt notation.

    """
    return _conventions.convert_3x3x3x3_to_6x6(S,[2.,4.])

def to_6x6_stiffness(C):
    """
    Convert symmetric 4th order stiffness tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    Returns
    -------
    C_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Voigt notation.

    """
    return _conventions.convert_3x3x3x3_to_6x6(C,[1.,1.])


def to_3x3_strain(epsilon_tilde):
    """
    Convert symmetric 2nd order strain tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    epsilon_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Voigt notation.

    Returns
    -------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    """
    return _conventions.convert_6_to_3x3(epsilon_tilde,.5)

def to_3x3_stress(sigma_tilde):
    """
    Convert symmetric 2nd order stress tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    sigma_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Voigt notation.

    Returns
    -------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    """
    return _conventions.convert_6_to_3x3(sigma_tilde,1.)


def to_3x3x3x3_compliance(S_tilde):
    """
    Convert symmetric 4th order compliance tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    S_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in Voigt notation.

    Returns
    -------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    """
    return _conventions.convert_6x6_to_3x3x3x3(S_tilde,[0.5,0.25])

def to_3x3x3x3_stiffness(C_tilde):
    """
    Convert symmetric 4th order stiffness tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    C_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Voigt notation.

    Returns
    -------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    """
    return _conventions.convert_6x6_to_3x3x3x3(C_tilde,[1.,1.])


def rotate_strain(O,epsilon_tilde):
    """
    Rotate symmetric 2nd order strain tensor in Voigt notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    epsilon_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Voigt notation.

    Returns
    -------
    epsilon_tilde' : numpy.ndarray of shape (...,6)
        Rotated 2nd order strain tensor in Voigt notation.

    """
    return _conventions.rotate(O,epsilon_tilde,[1.0,2.0],2)

def rotate_stress(O,sigma_tilde):
    """
    Rotate symmetric 2nd order stress tensor in Voigt notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    sigma_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Voigt notation.

    Returns
    -------
    sigma_tilde' : numpy.ndarray of shape (...,6)
        Rotated 2nd order stress tensor in Voigt notation.

    """
    return _conventions.rotate(O,sigma_tilde,[2.0,1.0],2)


def rotate_compliance(O,S_tilde):
    """
    Rotate symmetric 4th order compliance tensor in Voigt notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    S_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in Voigt notation.

    Returns
    -------
    S_tilde' : numpy.ndarray of shape (...,6,6)
        Rotated 4th order compliance tensor in Voigt notation.

    """
    return _conventions.rotate(O,S_tilde,[1.0,2.0],4)

def rotate_stiffness(O,C_tilde):
    """
    Rotate symmetric 4th order stiffness tensor in Voigt notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    C_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Voigt notation.

    Returns
    -------
    C_tilde' : numpy.ndarray of shape (...,6,6)
        Rotated 4th order stiffness tensor in Voigt notation.

    """
    return _conventions.rotate(O,C_tilde,[2.0,1.0],4)
