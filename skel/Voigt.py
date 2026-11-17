# SPDX-License-Identifier: AGPL-3.0-or-later
r"""
Compressed notation of symmetric tensors using convention attributed to W. Voigt.

Notes
-----
The following order is used to store the unique elements of a
2nd order tensor T (3x3 matrix) as a vector t of length 6.

.. math::

    \vb{t}_\text{Voigt} = (T_{xx}, T_{yy}, T_{zz}, w T_{yz}, w T_{xz}, w T_{xy})


Different weights are used for the representation of stress
(:math:`w = 1`) and strain (:math:`w = 2`).

References
----------
P. Helnwein, Computer methods in applied mechanics and engineering 190:2753-2770, 2001
https://doi.org/10.1016/s0045-7825(00)00263-2

T. Mánik, European Journal of Mechanics - A/Solids 90:104357, 2021
https://doi.org/10.1016/j.euromechsol.2021.104357

W. Voigt, Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse (2):117–176, 1900
https://gdz.sub.uni-goettingen.de/id/PPN252457811_1900
"""

import numpy as _np

from . import _conventions


# Basis of the Voigt notation for stress/stiffness (w=1) and strain/compliance (w=2).
# The two bases are dual to each other.
BASIS_STRESS = _np.zeros((6,3,3))
BASIS_STRESS[0,0,0] = BASIS_STRESS[1,1,1] = BASIS_STRESS[2,2,2] = 1.0
BASIS_STRESS[3,1,2] = BASIS_STRESS[3,2,1] = 0.5
BASIS_STRESS[4,0,2] = BASIS_STRESS[4,2,0] = 0.5
BASIS_STRESS[5,0,1] = BASIS_STRESS[5,1,0] = 0.5

BASIS_STRAIN = BASIS_STRESS.copy()
BASIS_STRAIN[3:] *= 2.0


def to_6_strain(epsilon):
    """
    Convert symmetric 2nd order strain tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    Returns
    -------
    epsilon_Voigt: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Voigt notation.

    """
    return _conventions.convert_3x3_to_6(epsilon,BASIS_STRAIN)

def to_6_stress(sigma):
    """
    Convert symmetric 2nd order stress tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    Returns
    -------
    sigma_Voigt: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Voigt notation.

    """
    return _conventions.convert_3x3_to_6(sigma,BASIS_STRESS)


def to_6x6_compliance(S):
    """
    Convert symmetric 4th order compliance tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    Returns
    -------
    S_Voigt: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in Voigt notation.

    """
    return _conventions.convert_3x3x3x3_to_6x6(S,BASIS_STRAIN)

def to_6x6_stiffness(C):
    """
    Convert symmetric 4th order stiffness tensor in matrix notation to Voigt notation.

    Parameters
    ----------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    Returns
    -------
    C_Voigt: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Voigt notation.

    """
    return _conventions.convert_3x3x3x3_to_6x6(C,BASIS_STRESS)


def to_3x3_strain(epsilon_Voigt):
    """
    Convert symmetric 2nd order strain tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    epsilon_Voigt: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Voigt notation.

    Returns
    -------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    """
    return _conventions.convert_6_to_3x3(epsilon_Voigt,BASIS_STRESS)

def to_3x3_stress(sigma_Voigt):
    """
    Convert symmetric 2nd order stress tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    sigma_Voigt: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Voigt notation.

    Returns
    -------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    """
    return _conventions.convert_6_to_3x3(sigma_Voigt,BASIS_STRAIN)


def to_3x3x3x3_compliance(S_Voigt):
    """
    Convert symmetric 4th order compliance tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    S_Voigt: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in Voigt notation.

    Returns
    -------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    """
    return _conventions.convert_6x6_to_3x3x3x3(S_Voigt,BASIS_STRESS)

def to_3x3x3x3_stiffness(C_Voigt):
    """
    Convert symmetric 4th order stiffness tensor in Voigt notation to matrix notation.

    Parameters
    ----------
    C_Voigt: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Voigt notation.

    Returns
    -------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    """
    return _conventions.convert_6x6_to_3x3x3x3(C_Voigt,BASIS_STRAIN)


def rotate_strain(r,epsilon_Voigt):
    """
    Rotate symmetric 2nd order strain tensor in Voigt notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    epsilon_Voigt: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in Voigt notation.

    Returns
    -------
    epsilon_Voigt : numpy.ndarray of shape (...,6)
        Rotated 2nd order strain tensor in Voigt notation.

    """
    return _conventions.rotate(r,epsilon_Voigt,BASIS_STRAIN,BASIS_STRESS,2)

def rotate_stress(r,sigma_Voigt):
    """
    Rotate symmetric 2nd order stress tensor in Voigt notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    sigma_Voigt: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in Voigt notation.

    Returns
    -------
    sigma_Voigt : numpy.ndarray of shape (...,6)
        Rotated 2nd order stress tensor in Voigt notation.

    """
    return _conventions.rotate(r,sigma_Voigt,BASIS_STRESS,BASIS_STRAIN,2)


def rotate_compliance(r,S_Voigt):
    """
    Rotate symmetric 4th order compliance tensor in Voigt notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    S_Voigt: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in Voigt notation.

    Returns
    -------
    S_Voigt : numpy.ndarray of shape (...,6,6)
        Rotated 4th order compliance tensor in Voigt notation.

    """
    return _conventions.rotate(r,S_Voigt,BASIS_STRAIN,BASIS_STRESS,4)

def rotate_stiffness(r,C_Voigt):
    """
    Rotate symmetric 4th order stiffness tensor in Voigt notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    C_Voigt: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in Voigt notation.

    Returns
    -------
    C_Voigt : numpy.ndarray of shape (...,6,6)
        Rotated 4th order stiffness tensor in Voigt notation.

    """
    return _conventions.rotate(r,C_Voigt,BASIS_STRESS,BASIS_STRAIN,4)
