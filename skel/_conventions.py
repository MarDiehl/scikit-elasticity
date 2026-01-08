# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import damask
r"""
Compressed notation of symmetric tensors.

Notes
-----
The following order is used to store the unique elements of a
2nd order tensor (3x3 matrix) as a vector of length 6.

.. math::
    \vb{t}_\mathrm{Voigt} = (t_{xx}, t_{yy}, t_{zz}, t_{yz}, t_{xz}, t_{xy})
"""

def convert_3x3_to_6(t,w):
    """
    Convert symmetric 2nd order tensor in matrix notation to vector notation.

    Parameters
    ----------
    t : numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.
    w : float
        Weighting factor for shear components.

    Returns
    -------
    t_tilde: numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in vector notation.

    """
    return np.stack([  t[...,0,0],
                       t[...,1,1],
                       t[...,2,2],
                     w*t[...,1,2],
                     w*t[...,0,2],
                     w*t[...,0,1]],-1)

def convert_6_to_3x3(t_tilde,w):
    """
    Convert symmetric 2nd order tensor in vector notation to matrix notation.

    Parameters
    ----------
    t_tilde : numpy.ndarray of shape (...,6)
        Symmetric 2nd order tensor in vector notation.
    w : float
        Weighting factor for shear components.

    Returns
    -------
    t: numpy.ndarray of shape (...,3,3)
        Symmetric 2nd order tensor in matrix notation.

    """
    return np.stack([np.stack([  t_tilde[...,0], w*t_tilde[...,5], w*t_tilde[...,4]],-1),
                     np.stack([w*t_tilde[...,5],   t_tilde[...,1], w*t_tilde[...,3]],-1),
                     np.stack([w*t_tilde[...,4], w*t_tilde[...,3],   t_tilde[...,2]],-1)],-2)


def convert_3x3x3x3_to_6x6(T,w):
    """
    Convert symmetric 4th order tensor in matrix notation to vector notation.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.
    w : float
        Weighting factor for shear components.

    Returns
    -------
    T_tilde: numpy.ndarray of shape (...,6,6)
        Symmetric 2nd order tensor in vector notation.

    """
    def assemble(t,idx):
        return np.stack([np.stack([t[...,idx[0][0],idx[1][0],i,j] for i,j in zip(idx[2],idx[3])],-1),
                         np.stack([t[...,idx[0][1],idx[1][1],i,j] for i,j in zip(idx[2],idx[3])],-1),
                         np.stack([t[...,idx[0][2],idx[1][2],i,j] for i,j in zip(idx[2],idx[3])],-1)],-2)

    a_11 = assemble(T, [[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
    a_12 = assemble(T, [[0,1,2],[0,1,2],[1,0,0],[2,2,1]])*w[0]
    a_21 = assemble(T, [[1,0,0],[2,2,1],[0,1,2],[0,1,2]])*w[0]
    a_22 = assemble(T, [[1,0,0],[2,2,1],[1,0,0],[2,2,1]])*w[1]

    return np.block([[a_11, a_12],
                     [a_21, a_22]])

def convert_6x6_to_3x3x3x3(T_tilde,w):
    """
    Convert symmetric 4th order tensor in vector notation to matrix notation.

    Parameters
    ----------
    T_tilde : numpy.ndarray of shape (...,6,6)
        Symmetric 2nd order tensor in vector notation.
    w : float
        Weighting factor for shear components.

    Returns
    -------
    T: numpy.ndarray of shape (...,3,3,3,3)
        Symmetric 4th order tensor in matrix notation.

    """
    def assemble(t,w_,i):
        return np.stack([np.stack([w_[0]*t[...,i,0],w_[1]*t[...,i,5],w_[1]*t[...,i,4]],-1),
                         np.stack([w_[1]*t[...,i,5],w_[0]*t[...,i,1],w_[1]*t[...,i,3]],-1),
                         np.stack([w_[1]*t[...,i,4],w_[1]*t[...,i,3],w_[0]*t[...,i,2]],-1)],-2)

    v_11 = assemble(T_tilde, [1.,w[0]],0)
    v_22 = assemble(T_tilde, [1.,w[0]],1)
    v_33 = assemble(T_tilde, [1.,w[0]],2)
    v_23 = v_32 = assemble(T_tilde,w,3)
    v_13 = v_31 = assemble(T_tilde,w,4)
    v_12 = v_21 = assemble(T_tilde,w,5)

    return np.stack([np.stack([v_11,v_12,v_13],-3),
                     np.stack([v_21,v_22,v_23],-3),
                     np.stack([v_31,v_32,v_33],-3)],-4)


def rotate(O,x_tilde,w,order):
    """
    Rotate symmetric 2nd or 4th order tensor in vector notation.

    Parameters
    ----------
    O: damask.Orientation of shape (...)
        Orientations.
    x_tilde: numpy.ndarray of shape (...,6) or (...,6,6)
        Symmetric 2nd or 4th order in vector notation.
    w: list of floats of shape (2)
        Weights
    order: integer [2,4]
        Order of the tensor to rotate.

    Returns
    -------
    x_tilde' : numpy.ndarray of shape (...,6)
        Rotated 2nd or 4th order strain tensor in vector notation.

    References
    ----------
    | https://scicomp.stackexchange.com/questions/35600
    | https://doi.org/10.1093/gji/ggy445

    """

    R = O.as_matrix()

    A_11 = np.stack([
        np.stack([R[...,0,0]**2, R[...,0,1]**2, R[...,0,2]**2],-1),
        np.stack([R[...,1,0]**2, R[...,1,1]**2, R[...,1,2]**2],-1),
        np.stack([R[...,2,0]**2, R[...,2,1]**2, R[...,2,2]**2],-1)],-2)

    A_22 = np.stack([
        np.stack([R[...,1,1]*R[...,2,2]+R[...,1,2]*R[...,2,1],
                  R[...,1,0]*R[...,2,2]+R[...,1,2]*R[...,2,0],
                  R[...,1,0]*R[...,2,1]+R[...,1,1]*R[...,2,0]],-1),
        np.stack([R[...,0,1]*R[...,2,2]+R[...,0,2]*R[...,2,1],
                  R[...,0,0]*R[...,2,2]+R[...,0,2]*R[...,2,0],
                  R[...,0,0]*R[...,2,1]+R[...,0,1]*R[...,2,0]],-1),
        np.stack([R[...,0,1]*R[...,1,2]+R[...,0,2]*R[...,1,1],
                  R[...,0,0]*R[...,1,2]+R[...,0,2]*R[...,1,0],
                  R[...,0,0]*R[...,1,1]+R[...,0,1]*R[...,1,0]],-1)],-2)

    A_12 = np.stack([
        np.stack([R[...,0,1]*R[...,0,2], R[...,0,0]*R[...,0,2], R[...,0,0]*R[...,0,1]],-1),
        np.stack([R[...,1,1]*R[...,1,2], R[...,1,0]*R[...,1,2], R[...,1,0]*R[...,1,1]],-1),
        np.stack([R[...,2,1]*R[...,2,2], R[...,2,0]*R[...,2,2], R[...,2,0]*R[...,2,1]],-1)],-2)*w[0]

    A_21 = np.stack([
        np.stack([R[...,1,0]*R[...,2,0], R[...,1,1]*R[...,2,1], R[...,1,2]*R[...,2,2]],-1),
        np.stack([R[...,0,0]*R[...,2,0], R[...,0,1]*R[...,2,1], R[...,0,2]*R[...,2,2]],-1),
        np.stack([R[...,0,0]*R[...,1,0], R[...,0,1]*R[...,1,1], R[...,0,2]*R[...,1,2]],-1)],-2)*w[1]

    blend = damask.util.shapeblender(O.shape,x_tilde.shape[:-order//2])
    s_op = damask.util.shapeshifter(O.shape,blend,mode='right') + (6,6)
    s_x = damask.util.shapeshifter(x_tilde.shape[:-order//2],blend,mode='left') + ((6,) if order == 2 else (6,6))

    op = np.broadcast_to(np.block([[A_11,A_12],
                                   [A_21,A_22]]).reshape(s_op),blend+(6,6))

    return np.einsum('...ij,...j',op,np.broadcast_to(x_tilde.reshape(s_x),blend+(6,))) if order == 2 else \
           np.einsum('...ij,...jk,...lk',op,np.broadcast_to(x_tilde.reshape(s_x),blend+(6,6)),op)
