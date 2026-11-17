# SPDX-License-Identifier: AGPL-3.0-or-later
r"""
Compressed notation of symmetric tensors using a natural basis.

Notes
-----
The following order is used to store the unique elements of a
2nd order tensor T (3x3 matrix) as a vector t of length 6.

.. math::

    \vb{t}_\text{natural} = (t_1, t_2, t_3, t_4, t_5, t_6)

The components are

.. math::

    t_1 &= \frac{T_{xx}+T_{yy}+T_{zz}}{\sqrt{3}} \\
    t_2 &= \frac{2T_{zz}-T_{xx}-T_{yy}}{\sqrt{6}}\cos\theta
           + \frac{T_{xx}-T_{yy}}{\sqrt{2}}\sin\theta \\
    t_3 &= \frac{2T_{zz}-T_{xx}-T_{yy}}{\sqrt{6}}\sin\theta
           - \frac{T_{xx}-T_{yy}}{\sqrt{2}}\cos\theta \\
    t_4 &= \sqrt{2}T_{yz} \\
    t_5 &= \sqrt{2}T_{xz} \\
    t_6 &= \sqrt{2}T_{xy}

The first component is the hydrostatic part, the next two components span
the deviatoric plane, and the last three are the shear components. The
optional parameter :math:`\theta` rotates the deviatoric plane, i.e. it mixes
:math:`t_2` and :math:`t_3`.

Because the underlying basis is orthonormal, the Euclidean norm of the
vector equals the Frobenius norm of the tensor.

References
----------
T. Mánik, European Journal of Mechanics - A/Solids 90:104357, 2021
https://doi.org/10.1016/j.euromechsol.2021.104357

U.F. Kocks et al., Texture and Anisotropy: Preferred Orientations in Polycrystals and their Effect on Materials Properties.
Cambridge University Press 1998, page 287

P. Lequeu, Comparison of crystallographic and continuum yield surfaces for textured polycrystals.
PhD thesis, McGill University, Montreal, Canada, 1986
https://escholarship.mcgill.ca/concern/theses/nk322f063

P. Van Houtte, Texture, Stress, and Microstructure 8(1):313-350, 1988
https://doi.org/10.1155/TSM.8-9.313
"""

import numpy as _np

from . import _conventions


def _basis(theta):
    """
    Orthonormal basis of the natural notation.

    Parameters
    ----------
    theta : float
        Rotation angle (in radians) of the deviatoric plane.

    Returns
    -------
    E : numpy.ndarray of shape (6,3,3)
        Basis tensors of the natural notation.

    """
    E1 = _np.diag([ 1.0, 1.0, 1.0])/_np.sqrt(3.0)
    E2 = _np.diag([-1.0,-1.0, 2.0])/_np.sqrt(6.0)
    E3 = _np.diag([-1.0, 1.0, 0.0])/_np.sqrt(2.0)
    E4 = _np.zeros((3,3)); E4[1,2] = E4[2,1] = 1.0/_np.sqrt(2.0)
    E5 = _np.zeros((3,3)); E5[0,2] = E5[2,0] = 1.0/_np.sqrt(2.0)
    E6 = _np.zeros((3,3)); E6[0,1] = E6[1,0] = 1.0/_np.sqrt(2.0)

    cos,sin = _np.cos(theta),_np.sin(theta)

    return _np.stack([E1, cos*E2-sin*E3, sin*E2+cos*E3, E4, E5, E6])


def to_6(t,theta=0.0):
    """
    Convert symmetric 2nd order tensor in matrix notation to natural notation.

    Parameters
    ----------
    t : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    t_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in natural notation.

    Notes
    -----
    Due to the definition of the natural notation, this
    function can be used to convert strain and stress tensors.

    """
    return _conventions.convert_3x3_to_6(t,_basis(theta))


def to_6_strain(epsilon,theta=0.0):
    """
    Convert symmetric 2nd order strain tensor in matrix notation to natural notation.

    Parameters
    ----------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    epsilon_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in natural notation.

    """
    return to_6(epsilon,theta)


def to_6_stress(sigma,theta=0.0):
    """
    Convert symmetric 2nd order stress tensor in matrix notation to natural notation.

    Parameters
    ----------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    sigma_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in natural notation.

    """
    return to_6(sigma,theta)


def to_6x6(T,theta=0.0):
    """
    Convert symmetric 4th order tensor in matrix notation to natural notation.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    T_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in natural notation.

    Notes
    -----
    Due to the definition of the natural notation, this
    function can be used to convert compliance and stiffness tensors.

    """
    return _conventions.convert_3x3x3x3_to_6x6(T,_basis(theta))


def to_6x6_compliance(S,theta=0.0):
    """
    Convert symmetric 4th order compliance tensor in matrix notation to natural notation.

    Parameters
    ----------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    S_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in natural notation.

    """
    return to_6x6(S,theta)


def to_6x6_stiffness(C,theta=0.0):
    """
    Convert symmetric 4th order stiffness tensor in matrix notation to natural notation.

    Parameters
    ----------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    C_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in natural notation.

    """
    return to_6x6(C,theta)


def to_3x3(t_natural,theta=0.0):
    """
    Convert symmetric 2nd order tensor in natural notation to matrix notation.

    Parameters
    ----------
    t_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    t : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.

    Notes
    -----
    Due to the definition of the natural notation, this
    function can be used to convert strain and stress tensors.

    """
    return _conventions.convert_6_to_3x3(t_natural,_basis(theta))


def to_3x3_strain(epsilon_natural,theta=0.0):
    """
    Convert symmetric 2nd order strain tensor in natural notation to matrix notation.

    Parameters
    ----------
    epsilon_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    epsilon : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order strain tensor in matrix notation.

    """
    return to_3x3(epsilon_natural,theta)


def to_3x3_stress(sigma_natural,theta=0.0):
    """
    Convert symmetric 2nd order stress tensor in natural notation to matrix notation.

    Parameters
    ----------
    sigma_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    sigma : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order stress tensor in matrix notation.

    """
    return to_3x3(sigma_natural,theta)


def to_3x3x3x3(T_natural,theta=0.0):
    """
    Convert symmetric 4th order tensor in natural notation to matrix notation.

    Parameters
    ----------
    T_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    T : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.

    Notes
    -----
    Due to the definition of the natural notation, this
    function can be used to convert compliance and stiffness tensors.

    """
    return _conventions.convert_6x6_to_3x3x3x3(T_natural,_basis(theta))


def to_3x3x3x3_compliance(S_natural,theta=0.0):
    """
    Convert symmetric 4th order compliance tensor in natural notation to matrix notation.

    Parameters
    ----------
    S_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    S : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order compliance tensor in matrix notation.

    """
    return to_3x3x3x3(S_natural,theta)


def to_3x3x3x3_stiffness(C_natural,theta=0.0):
    """
    Convert symmetric 4th order stiffness tensor in natural notation to matrix notation.

    Parameters
    ----------
    C_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    C : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order stiffness tensor in matrix notation.

    """
    return to_3x3x3x3(C_natural,theta)


def _rotate(r,x_natural,order,theta=0.0):
    """
    Rotate symmetric 2nd or 4th order tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    x_natural: numpy.ndarray of shape (...,6) or (...,6,6)
        Symmetric 2nd or 4th order tensor in natural notation.
    order: integer [2,4]
        Order of the tensor to rotate.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    x_natural: numpy.ndarray of shape (...,6) or (...,6,6)
        Rotated 2nd or 4th order tensor in natural notation.

    """
    E = _basis(theta)
    return _conventions.rotate(r,x_natural,E,E,order)


def rotate_2nd(r,t_natural,theta=0.0):
    """
    Rotate symmetric 2nd order tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    t_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    t_natural : numpy.ndarray of shape (...,6)
        Rotated 2nd order tensor in natural notation.

    Notes
    -----
    Due to the definition of the natural notation, this
    function can be used to rotate strain and stress tensors.

    """
    return _rotate(r,t_natural,2,theta)


def rotate_strain(r,epsilon_natural,theta=0.0):
    """
    Rotate symmetric 2nd order strain tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    epsilon_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order strain tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    epsilon_natural : numpy.ndarray of shape (...,6)
        Rotated 2nd order strain tensor in natural notation.

    """
    return rotate_2nd(r,epsilon_natural,theta)


def rotate_stress(r,sigma_natural,theta=0.0):
    """
    Rotate symmetric 2nd order stress tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    sigma_natural: numpy.ndarray of shape (...,6)
        Symmetric 2nd order stress tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    sigma_natural : numpy.ndarray of shape (...,6)
        Rotated 2nd order stress tensor in natural notation.

    """
    return rotate_2nd(r,sigma_natural,theta)


def rotate_4th(r,T_natural,theta=0.0):
    """
    Rotate symmetric 4th order tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    T_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    T_natural : numpy.ndarray of shape (...,6,6)
        Rotated 4th order tensor in natural notation.

    Notes
    -----
    Due to the definition of the natural notation, this
    function can be used to rotate compliance and stiffness tensors.

    """
    return _rotate(r,T_natural,4,theta)


def rotate_compliance(r,S_natural,theta=0.0):
    """
    Rotate symmetric 4th order compliance tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    S_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order compliance tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    S_natural : numpy.ndarray of shape (...,6,6)
        Rotated 4th order compliance tensor in natural notation.

    """
    return rotate_4th(r,S_natural,theta)


def rotate_stiffness(r,C_natural,theta=0.0):
    """
    Rotate symmetric 4th order stiffness tensor in natural notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Orientations.
    C_natural: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order stiffness tensor in natural notation.
    theta : float, optional
        Rotation angle (in radians) of the deviatoric plane. Defaults to 0.

    Returns
    -------
    C_natural : numpy.ndarray of shape (...,6,6)
        Rotated 4th order stiffness tensor in natural notation.

    """
    return rotate_4th(r,C_natural,theta)
