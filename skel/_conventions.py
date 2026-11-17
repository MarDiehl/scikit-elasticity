# SPDX-License-Identifier: AGPL-3.0-or-later
r"""
Compressed notation of symmetric tensors.

Notes
-----
All conventions store the unique elements of a symmetric 2nd order
tensor t (3x3 matrix) as a vector t' of length 6. The first three
components are associated with the normal components and the last three
components with the shear components. The precise mapping is encoded by
a basis :math:`E` of symmetric 2nd order tensors,

.. math::
    t'_a = E_a : t = \sum_{ij} E_{a,ij} t_{ij},

and the inverse conversion uses the dual basis :math:`F`,

.. math::
    t = \sum_a t'_a F_a, \qquad E_a : F_b = \delta_{ab}.

For an orthonormal basis, the dual basis equals the basis itself but
for non-orthonormal basis the distinction matters.

References
----------
P. Helnwein, Computer methods in applied mechanics and engineering 190:2753-2770, 2001
https://doi.org/10.1016/s0045-7825(00)00263-2

T. Mánik, European Journal of Mechanics - A/Solids 90:104357, 2021
https://doi.org/10.1016/j.euromechsol.2021.104357
"""

import numpy as np
import damask


def convert_3x3_to_6(t,E):
    """
    Convert symmetric 2nd order tensor in matrix notation to vector notation.

    Parameters
    ----------
    t : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.
    E : numpy.ndarray of shape (6,3,3)
        Basis tensors of the vector notation.

    Returns
    -------
    t_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in vector notation.

    """
    return t.reshape(t.shape[:-2]+(9,)) @ E.reshape(6,9).T


def convert_6_to_3x3(t_tilde,F):
    """
    Convert symmetric 2nd order tensor in vector notation to matrix notation.

    Parameters
    ----------
    t_tilde : numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in vector notation.
    F : numpy.ndarray of shape (6,3,3)
        Dual basis tensors of the vector notation, i.e. the basis used
        for reconstruction.

    Returns
    -------
    t: numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.

    """
    return (t_tilde @ F.reshape(6,9)).reshape(t_tilde.shape[:-1]+(3,3))


def convert_3x3x3x3_to_6x6(T,E):
    """
    Convert symmetric 4th order tensor in matrix notation to vector notation.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.
    E : numpy.ndarray of shape (6,3,3)
        Basis tensors of the vector notation.

    Returns
    -------
    T_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in vector notation.

    """
    E_ = E.reshape(6,9)
    return (E_ @ T.reshape(T.shape[:-4]+(9,9))) @ E_.T


def convert_6x6_to_3x3x3x3(T_tilde,F):
    """
    Convert symmetric 4th order tensor in vector notation to matrix notation.

    Parameters
    ----------
    T_tilde : numpy.ndarray of shape (...,6,6)
        Symmetric 4th order tensor in vector notation.
    F : numpy.ndarray of shape (6,3,3)
        Dual basis tensors of the vector notation, i.e. the basis used
        for reconstruction.

    Returns
    -------
    T: numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.

    """
    F_ = F.reshape(6,9)
    return (F_.T @ T_tilde @ F_).reshape(T_tilde.shape[:-2]+(3,3,3,3))


def rotate(r,x_tilde,E,F,order):
    """
    Rotate symmetric 2nd or 4th order tensor in vector notation.

    Parameters
    ----------
    r: damask.Rotation of shape (...)
        Rotations to apply. Must be broadcastable.
    x_tilde: numpy.ndarray of shape (...,6) or (...,6,6)
        Symmetric 2nd or 4th order in vector notation.
    E : numpy.ndarray of shape (6,3,3)
        Basis tensors of the vector notation.
    F : numpy.ndarray of shape (6,3,3)
        Dual basis tensors of the vector notation.
    order: integer [2,4]
        Order of the tensor to rotate.

    Returns
    -------
    x_tilde: numpy.ndarray of shape (...,6) or (...,6,6)
        Rotated 2nd or 4th order tensor in vector notation.

    References
    ----------
    | https://scicomp.stackexchange.com/questions/35600
    | https://doi.org/10.1093/gji/ggy445

    """
    R = r.as_matrix()

    # rotation matrix in the vector notation: Q_ab = E_a : (R F_b R^T)
    # with (R F_b R^T)_ij = R_ik F_b,kl R_jl
    S = np.einsum('...ik,...jl->...ijkl',R,R).reshape(R.shape[:-2]+(9,9))
    Q = E.reshape(6,9) @ S @ F.reshape(6,9).T

    blend = damask.util.shapeblender(r.shape,x_tilde.shape[:-order//2])
    s_op = damask.util.shapeshifter(r.shape,blend,mode='right') + (6,6)
    s_x = damask.util.shapeshifter(x_tilde.shape[:-order//2],blend,mode='left') + ((6,) if order == 2 else (6,6))

    op = np.broadcast_to(Q.reshape(s_op),blend+(6,6))

    return np.einsum('...ij,...j',op,np.broadcast_to(x_tilde.reshape(s_x),blend+(6,))) if order == 2 else \
           np.einsum('...ij,...jk,...lk',op,np.broadcast_to(x_tilde.reshape(s_x),blend+(6,6)),op)
