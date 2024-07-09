# SPDX-License-Identifier: AGPL-3.0-or-later
from . import _conventions
"""
Convention attributed to J. Mandel for the compressed notation of symmetric tensors.

Notes
-----
The following order is used to store the unique elements of a
2nd order tensor (3x3 matrix) as a vector of length 6.

.. math::
    \vb{t}_\mathrm{Voigt} = (t_{xx}, t_{yy}, t_{zz}, t_{yz}, t_{xz}, t_{xy})


The same weight is used for the representation of stress and strain.
"""

def to_6(t):
    """
    Convert symmetric 2nd order tensor in matrix notation to Mandel notation.

    Parameters
    ----------
    t : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.

    Returns
    -------
    t_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in Mandel notation.

    Notes
    -----
    Due to the definition of the Mandel notation, this
    function can be used to convert strain and stress tensors.

    """
    return _conventions.convert_3x3_to_6(t,2.**.5)

def to_6_strain(epsilon):
    """
    Convert symmetric 2nd order strain tensor in matrix notation to Mandel notation.

    Parameters
    ----------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    Returns
    -------
    epsilon_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Mandel notation.

    """
    return to_6(epsilon)

def to_6_stress(sigma):
    """
    Convert symmetric 2nd order stress tensor in matrix notation to Mandel notation.

    Parameters
    ----------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    Returns
    -------
    sigma_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Mandel notation.

    """
    return to_6(sigma)


def to_6x6(T):
    """
    Convert symmetric 4th order tensor in matrix notation to Mandel notation.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.

    Returns
    -------
    T_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in Mandel notation.

    Notes
    -----
    Due to the definition of the Mandel notation, this
    function can be used to convert compliance and stiffness tensors.

    """
    return _conventions.convert_3x3x3x3_to_6x6(T,[2.**.5,2.])

def to_6x6_compliance(S):
    """
    Convert symmetric 4th order compliance tensor in matrix notation to Mandel notation.

    Parameters
    ----------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    Returns
    -------
    S_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in Mandel notation.

    """
    return to_6x6(S)

def to_6x6_stiffness(C):
    """
    Convert symmetric 4th order stiffness tensor in matrix notation to Mandel notation.

    Parameters
    ----------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    Returns
    -------
    C_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Mandel notation.

    """
    return to_6x6(C)


def to_3x3(t_tilde):
    """
    Convert symmetric 2nd order tensor in Mandel notation to matrix notation.

    Parameters
    ----------
    t_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in Mandel notation.

    Returns
    -------
    t : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.

    Notes
    -----
    Due to the definition of the Mandel notation, this
    function can be used to convert strain and stress tensors.

    """
    return _conventions.convert_6_to_3x3(t_tilde,2.**(-.5))

def to_3x3_strain(epsilon_tilde):
    """
    Convert symmetric 2nd order strain tensor in Mandel notation to matrix notation.

    Parameters
    ----------
    epsilon_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Mandel notation.

    Returns
    -------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    """
    return to_3x3(epsilon_tilde)

def to_3x3_stress(sigma_tilde):
    """
    Convert symmetric 2nd order stress tensor in Mandel notation to matrix notation.

    Parameters
    ----------
    sigma_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Mandel notation.

    Returns
    -------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    """
    return to_3x3(sigma_tilde)


def to_3x3x3x3(T_tilde):
    """
    Convert symmetric 4th order tensor in Mandel notation to matrix notation.

    Parameters
    ----------
    T_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in Mandel notation.

    Returns
    -------
    T : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.

    Notes
    -----
    Due to the definition of the Mandel notation, this
    function can be used to convert compliance and stiffness tensors.

    """
    return _conventions.convert_6x6_to_3x3x3x3(T_tilde,[2.**(-.5),0.5])

def to_3x3x3x3_compliance(S_tilde):
    """
    Convert symmetric 4th order compliance tensor in Mandel notation to matrix notation.

    Parameters
    ----------
    S_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in Mandel notation.

    Returns
    -------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    """
    return to_3x3x3x3(S_tilde)

def to_3x3x3x3_stiffness(C_tilde):
    """
    Convert symmetric 4th order stiffness tensor in Mandel notation to matrix notation.

    Parameters
    ----------
    C_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Mandel notation.

    Returns
    -------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    """
    return to_3x3x3x3(C_tilde)


def rotate_2nd(O,t_tilde):
    """
    Rotate symmetric 2nd order tensor in Mandel notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    t_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in Mandel notation.

    Returns
    -------
    t_tilde' : numpy.ndarray of shape (...,6)
        Rotated 2nd order tensor in Mandel notation.

    Notes
    -----
    Due to the definition of the Mandel notation, this
    function can be used to rotate strain and stress tensors.

    """
    return _conventions.rotate(O,t_tilde,[2.**.5,2.**.5],2)

def rotate_strain(O,epsilon_tilde):
    """
    Rotate symmetric 2nd order strain tensor in Mandel notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    epsilon_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Mandel notation.

    Returns
    -------
    epsilon_tilde' : numpy.ndarray of shape (...,6)
        Rotated 2nd order strain tensor in Mandel notation.

    """
    return rotate_2nd(O,epsilon_tilde)

def rotate_stress(O,sigma_tilde):
    """
    Rotate symmetric 2nd order stress tensor in Mandel notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    sigma_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Mandel notation.

    Returns
    -------
    sigma_tilde' : numpy.ndarray of shape (...,6)
        Rotated 2nd order stress tensor in Mandel notation.

    """
    return rotate_2nd(O,sigma_tilde)


def rotate_4th(O,T_tilde):
    """
    Rotate symmetric 4th order tensor in Mandel notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    T_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in Mandel notation.

    Returns
    -------
    T_tilde' : numpy.ndarray of shape (...,6,6)
        Rotated 4th order tensor in Mandel notation.

    Notes
    -----
    Due to the definition of the Mandel notation, this
    function can be used to rotate compliance and stiffness tensors.

    """
    return _conventions.rotate(O,T_tilde,[2.**.5,2.**.5],4)

def rotate_compliance(O,S_tilde):
    """
    Rotate symmetric 4th order compliance tensor in Mandel notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    S_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in Mandel notation.

    Returns
    -------
    S_tilde' : numpy.ndarray of shape (...,6,6)
        Rotated 4th order compliance tensor in Mandel notation.

    """
    return rotate_4th(O,S_tilde)

def rotate_stiffness(O,C_tilde):
    """
    Rotate symmetric 4th order stiffness tensor in Mandel notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    C_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Mandel notation.

    Returns
    -------
    C_tilde' : numpy.ndarray of shape (...,6,6)
        Rotated 4th order stiffness tensor in Mandel notation.

    """
    return rotate_4th(O,C_tilde)
