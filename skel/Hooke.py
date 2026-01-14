# SPDX-License-Identifier: AGPL-3.0-or-later
# https://dictionary.iucr.org/Crystal_system
# https://dictionary.iucr.org/Lattice_system
# https://dictionary.iucr.org/Crystal_family
import numpy as _np
from . import Voigt as _Voigt
"""
Linear elasticity according to the generalization of Hooke's law.

Notes
-----
Tensors are given in Voigt form (6x6 with different weights).
"""

def _ensure_equal(**kwargs):
    """
    Ensure that all arguments are equal.

    Special handling of None.
    """
    mode = kwargs.pop('mode')
    crystal_system = kwargs.pop('crystal_system')
    kwargs = {k:(float(v) if v is not None else v) for k,v in kwargs.items()}
    if len(v := set(kwargs.values())) == 1:
        return list(v)*len(kwargs)
    elif len(v := set(kwargs.values()) - {None}) == 1:
        return list(v)*len(kwargs)
    else:
        invalid = ', '.join([k.replace('X',mode) for k in kwargs.keys()])
        raise ValueError(f'{invalid} need to have the same value for {crystal_system} crystal system')

def _ensure_zero(**kwargs):
    """
    Ensure that all arguments are zero.

    Special handling of None.
    """
    mode = kwargs.pop('mode')
    crystal_system = kwargs.pop('crystal_system')
    kwargs = {k:(float(v) if v is not None else v) for k,v in kwargs.items()}
    valid = {None,0.0}
    if set(kwargs.values()).union(valid) != valid:
        invalid = ', '.join([k.replace('X',mode) for k,v in kwargs.items() if v not in valid])
        raise ValueError(f'{invalid} need to be 0.0 for {crystal_system} crystal system')
    return [0.0] * len(kwargs)


def _assemble(crystal_system, mode,
              X_11, X_12, X_13, X_14, X_15, X_16,
                    X_22, X_23, X_24, X_25, X_26,
                          X_33, X_34, X_35, X_36,
                                X_44, X_45, X_46,
                                      X_55, X_56,
                                            X_66):
    r"""
    Assemble stiffness or compliance matrix.

    Parameters
    ----------
    crystal_system
    mode
    Nye, table 9 (page 140)

    Notes
    -----
    isotropic:
      - X_11 = X_22 = X_33
      - X_12 = X_13 = X_23
      - X_44 = X_55 = X_66 = f×(X_11-X_12)
    cubic:
      - X_11 = X_22 = X_33
      - X_12 = X_13 = X_23
      - X_44 = X_55 = X_66
    hexagonal:
      - X_11 = X_22
      - X_13 = X_23
      - X_44 = X_55
      - X_66 = f×(X_11-X_12)
    trigonal:
      - X_11 = X_22
      - X_13 = X_23
      - X_44 = X_55
      - X_66 = f×(X_11-X_12)
      - X_14 = -X_24 = X_56
      - X_15 = -X_25 = -X_46 (or 0.0)
    tetragonal:
      - X_11 = X_22
      - X_13 = X_23
      - X_44 = X_55
      - X_16 = -X_26 (or 0.0)
    """
    match mode:
        case 'C':
            f_1, f_2 = 1.0, 0.5
        case 'S':
            f_1, f_2 = 2.0, 2.0
        case _:
            raise ValueError(f'invalid mode "{mode}"')


    match crystal_system:

        case 'isotropic':
            X_14,X_15,X_16, X_24,X_25,X_26, X_34,X_35,X_36, X_45,X_46, X_56 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                             X_14=X_14,X_15=X_15,X_16=X_16,
                             X_24=X_24,X_25=X_25,X_26=X_26,
                             X_34=X_34,X_35=X_35,X_36=X_36,
                                       X_45=X_45,X_46=X_46,
                                                 X_56=X_56)

            X_11,X_22,X_33 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                           X_11=X_11, X_22=X_22, X_33=X_33)
            X_12,X_13,X_23 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                           X_12=X_12, X_13=X_13, X_23=X_23)
            X_44,X_55,X_66 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                           X_44=X_44, X_55=X_55, X_66=X_66)

            if   X_11 is None and X_12 is not None and X_44 is not None:
                X_11=X_22=X_33 = X_44/f_2 + X_12
            elif X_12 is None and X_11 is not None and X_44 is not None:
                X_12=X_13=X_23 = X_11 - X_44/f_2
            elif X_44 is None and X_11 is not None and X_12 is not None:
                X_44=X_55=X_66 = f_2*(X_11-X_12)
            elif not _np.isclose(X_44,f_2*(X_11-X_12)):
                raise ValueError(f'condition {mode}_44 = {f_2}×({mode}_11 - {mode}_12) for isotropic crystal system violated')


        case 'cubic':
            X_14,X_15,X_16, X_24,X_25,X_26, X_34,X_35,X_36, X_45,X_46, X_56 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                             X_14=X_14,X_15=X_15,X_16=X_16,
                             X_24=X_24,X_25=X_25,X_26=X_26,
                             X_34=X_34,X_35=X_35,X_36=X_36,
                                       X_45=X_45,X_46=X_46,
                                                 X_56=X_56)

            X_11,X_22,X_33 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                           X_11=X_11, X_22=X_22, X_33=X_33)
            X_12,X_13,X_23 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                           X_12=X_12, X_13=X_13, X_23=X_23)
            X_44,X_55,X_66 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                           X_44=X_44, X_55=X_55, X_66=X_66)


        case 'hexagonal':
            X_14,X_15,X_16, X_24,X_25,X_26, X_34,X_35,X_36, X_45,X_46, X_56 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                             X_14=X_14,X_15=X_15,X_16=X_16,
                             X_24=X_24,X_25=X_25,X_26=X_26,
                             X_34=X_34,X_35=X_35,X_36=X_36,
                                       X_45=X_45,X_46=X_46,
                                                 X_56=X_56)

            X_11,X_22 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_11=X_11,X_22=X_22)
            X_13,X_23 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_13=X_13,X_23=X_23)
            X_44,X_55 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_44=X_44,X_55=X_55)

            if   X_11 is None and [X_12,X_66].count(None) == 0:
                X_11=X_22 = X_66/f_2 + X_12
            elif X_12 is None and [X_11,X_66].count(None) == 0:
                X_12 = X_11 - X_66/f_2
            elif X_66 is None and [X_11,X_12].count(None) == 0:
                X_66 = f_2*(X_11-X_12)
            elif not _np.isclose(X_66,f_2*(X_11-X_12)):
                raise ValueError(f'condition {mode}_66 = {f_2}×({mode}_11 - {mode}_12) for hexagonal crystal system violated')


        case 'trigonal':
            f = f'2×' if mode == 'S' else ''
            X_16,X_26,X_34,X_35,X_36,X_45 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                                                 X_16=X_16,
                                                 X_26=X_26,
                             X_34=X_34,X_35=X_35,X_36=X_36,
                                       X_45=X_45)

            X_11,X_22 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_11=X_11,X_22=X_22)
            X_13,X_23 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_13=X_13,X_23=X_23)
            X_44,X_55 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_44=X_44,X_55=X_55)

            if   X_11 is None and [X_12,X_66].count(None) == 0:
                X_11=X_22 = X_66/f_2 + X_12
            elif X_12 is None and [X_11,X_66].count(None) == 0:
                X_12 = X_11 - X_66/f_2
            elif X_66 is None and [X_11,X_12].count(None) == 0:
                X_66 = f_2*(X_11-X_12)
            elif not _np.isclose(X_66,f_2*(X_11-X_12)):
                raise ValueError(f'condition {mode}_66 = {f_2}×({mode}_11 - {mode}_12) for trigonal crystal system violated')

            msg = f'condition {mode}_14 = -{mode}_24 = {f}{mode}_56 for trigonal crystal system violated'
            match [X_14,X_24,X_56].count(None):
                case 3:
                    raise ValueError(msg)
                case 2:
                    if X_14 is None and X_24 is None:
                        X_14 = X_56/f_1
                        X_24 = -X_56/f_1
                    elif X_14 is None and X_56 is None:
                        X_14 = -X_24
                        X_56 = -X_24*f_1
                    else:
                        X_24 = -X_14
                        X_56 = X_14*f_1
                case 1:
                    if X_14 is None:
                        X_14 = -X_24
                        if X_24 != -X_56/f_1: raise ValueError(msg)
                    elif X_24 is None:
                        X_24 = X_14
                        if X_14 != X_56/f_1: raise ValueError(msg)
                    else:
                        X_56 = f_1*X_14
                        if X_14 != -X_24: raise ValueError(msg)
                case 0:
                    if X_14 != -X_24 or X_14 != f_1*X_56:
                        raise ValueError(msg)

            msg = f'condition {mode}_15 = -{mode}_25 = -{f}{mode}_46 for trigonal crystal system violated'
            match [X_15,X_25,X_46].count(None):
                case 3:
                    X_15=X_25=X_46 = 0.0
                case 2:
                    if X_15 is None and X_25 is None:
                        X_15 = X_46/f_1
                        X_25 = -X_46/f_1
                    elif X_15 is None and X_46 is None:
                        X_15 = -X_25
                        X_46 = -X_25*f_1
                    else:
                        X_25 = -X_15
                        X_46 = X_15*f_1
                case 1:
                    if X_15 is None:
                        X_15 = -X_25
                        if X_25 != -X_46/f_1: raise ValueError(msg)
                    elif X_25 is None:
                        X_25 = X_15
                        if X_15 != X_46/f_1: raise ValueError(msg)
                    else:
                        X_46 = f_1*X_15
                        if X_15 != -X_25: raise ValueError(msg)
                case 0:
                    if X_15 != -X_25 or X_15 != -f_1*X_46:
                        raise ValueError(msg)


        case 'tetragonal':
            X_14,X_15, X_24,X_25, X_34,X_35,X_36, X_45,X_46, X_56 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                             X_14=X_14,X_15=X_15,
                             X_24=X_24,X_25=X_25,
                             X_34=X_34,X_35=X_35,X_36=X_36,
                                       X_45=X_45,X_46=X_46,
                                                 X_56=X_56)

            X_11,X_22 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_11=X_11,X_22=X_22)
            X_13,X_23 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_13=X_13,X_23=X_23)
            X_44,X_55 = _ensure_equal(crystal_system=crystal_system, mode=mode,
                                      X_44=X_44,X_55=X_55)

            if X_16 is None and X_26 is None:
                X_16 = X_26 = 0.0
            elif X_16 is None:
                X_16 = -X_26
            elif X_26 is None:
                X_26 = -X_16
            else:
                if X_16+X_26 != 0.0:
                    raise ValueError(f'condition {mode}_16 = -{mode}_16 for tetragonal crystal system violated')


        case 'orthorhombic':
            X_14,X_15,X_16, X_24,X_25,X_26, X_34,X_35,X_36, X_45,X_46, X_56 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                             X_14=X_14,X_15=X_15,X_16=X_16,
                             X_24=X_24,X_25=X_25,X_26=X_26,
                             X_34=X_34,X_35=X_35,X_36=X_36,
                                       X_45=X_45,X_46=X_46,
                                                 X_56=X_56)

        case 'monoclinic':
            X_14,X_16, X_24,X_26, X_34,X_36, X_45, X_56 = \
                _ensure_zero(crystal_system=crystal_system, mode=mode,
                             X_14=X_14,          X_16=X_16,
                             X_24=X_24,          X_26=X_26,
                             r_34=X_34,          X_36=X_36,
                                       X_45=X_45,
                                                 X_56=X_56)

        case 'triclinic':
            pass

        case _:
            raise ValueError(f'invalid crystal system {crystal_system}')

    upper = _np.array([[X_11, X_12, X_13, X_14, X_15, X_16],
                       [ 0,   X_22, X_23, X_24, X_25, X_26],
                       [ 0,   0,    X_33, X_34, X_35, X_36],
                       [ 0,   0,    0,    X_44, X_45, X_46],
                       [ 0,   0,    0,    0,    X_55, X_56],
                       [ 0,   0,    0,    0,    0,    X_66]])

    if _np.count_nonzero(idx := (upper == None).nonzero()) > 0:
        c = [f'{mode}_{r+1}{c+1}' for r,c in zip(idx[0],idx[1])]
        raise ValueError(f'{",".join(c)} need to be defined for {crystal_system} crystal system')

    return upper + _np.tril(upper.T,-1)


def C(crystal_system,*,
      C_11=None, C_12=None, C_13=None, C_14=None, C_15=None, C_16=None,
                 C_22=None, C_23=None, C_24=None, C_25=None, C_26=None,
                            C_33=None, C_34=None, C_35=None, C_36=None,
                                       C_44=None, C_45=None, C_46=None,
                                                  C_55=None, C_56=None,
                                                             C_66=None):
    """
    Assemble stiffness matrix.

    Entries are from the upper diagonal.
    """
    C = _assemble(crystal_system,'C', C_11, C_12, C_13, C_14, C_15, C_16,
                                            C_22, C_23, C_24, C_25, C_26,
                                                  C_33, C_34, C_35, C_36,
                                                        C_44, C_45, C_46,
                                                              C_55, C_56,
                                                                    C_66)
    if not stable(crystal_system,C=C):
        raise ValueError('unstable crystal')

    return C


def S(crystal_system,*,
      S_11=None, S_12=None, S_13=None, S_14=None, S_15=None, S_16=None,
                 S_22=None, S_23=None, S_24=None, S_25=None, S_26=None,
                            S_33=None, S_34=None, S_35=None, S_36=None,
                                       S_44=None, S_45=None, S_46=None,
                                                  S_55=None, S_56=None,
                                                             S_66=None):
    """
    Assemble compliance matrix.

    Entries are from the upper diagonal.
    """
    S = _assemble(crystal_system,'S', S_11, S_12, S_13, S_14, S_15, S_16,
                                            S_22, S_23, S_24, S_25, S_26,
                                                  S_33, S_34, S_35, S_36,
                                                        S_44, S_45, S_46,
                                                              S_55, S_56,
                                                                    S_66)
    if not stable(crystal_system,S=S):
        raise ValueError('unstable crystal')

    return S


def stable(crystal_system,*,C=None,S=None):
    """
    Determine whether a stiffness or compliance matrix is stable.

    Parameters
    ----------
    C: numpy.ndarray, shape (6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    F. Mouhat and F.-X. Coudert, Physical Review B 90:224104, 2014
    https://doi.org/10.1103/PhysRevB.90.224104
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')

    if C is None: C = _np.linalg.inv(S)

    match crystal_system:
        case 'isotropic':
            return C[0,0] > abs(C[0,1]) and \
                   C[0,0]+2*C[0,1] > 0
        case 'cubic':
            return C[0,0] > abs(C[0,1]) and \
                   C[0,0]+2*C[0,1] > 0 and \
                   C[3,3] > 0
        case 'hexagonal':
            return C[0,0] > abs(C[0,1]) and \
                   2*C[0,2]**2 < C[2,2]*(C[0,0]+C[0,1]) and \
                   C[3,3] > 0
        case 'trigonal':
            # Called rhombohedral in the reference, but seems Nye refers
            return C[0,0] > abs(C[0,1]) and \
                   C[3,3] > 0 and \
                   C[0,2]**2 < 0.5*C[2,2]*(C[0,0]+C[0,1]) and \
                   C[0,3]**2+C[0,4]**2 < 0.5*C[3,3]*(C[0,0]-C[0,1])
        case 'tetragonal':
            return C[0,0] > abs(C[0,1]) and \
                   2*C[0,2]**2 < C[2,2]*(C[0,0]+C[0,1]) and \
                   C[3,3] > 0 and \
                   2*C[0,5]**2 < C[5,5]*(C[0,0]-C[0,1])
        case 'orthorhombic':
            return C[0,0] > 0 and \
                   C[0,0]*C[1,1] > C[0,1]**2 and \
                   C[0,0]*C[1,1]*C[2,2]+2*C[0,1]*C[0,2]*C[1,2]-C[0,0]*C[1,2]**2-C[1,1]*C[0,2]**2-C[2,2]*C[0,1]**2 > 0 and \
                   C[3,3] > 0 and \
                   C[4,4] > 0 and \
                   C[5,5] > 0
        case 'monoclinic' | 'triclinic':
            return bool(_np.all(_np.linalg.eigvalsh(C)>0))
        case _:
            return ValueError(f'invalid crystal system {crystal_system}')


def E(x,*,C=None,S=None):
    """
    Directional Young's modulus.

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    https://mtex-toolbox.github.io/stiffnessTensor.YoungsModulus.html
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)

    return 1/_np.einsum('...ijkl,...i,...j,...k,...l',_Voigt.to_3x3x3x3_compliance(S),x,x,x,x)

def G(h,u,*,C=None,S=None):
    """
    Directional shear modulus.

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    https://mtex-toolbox.github.io/stiffnessTensor.shearModulus.html
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)

    return .25/_np.einsum('...ijkl,...i,...j,...k,...l',_Voigt.to_3x3x3x3_compliance(S),h,u,h,u)

def nu(x,y,*,C=None,S=None):
    """
    Directional Poisson ratio.

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    https://mtex-toolbox.github.io/stiffnessTensor.PoissonRatio.html
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)

    S_ = _Voigt.to_3x3x3x3_compliance(S)
    return -_np.einsum('...ijkl,...i,...j,...k,...l',S_,x,x,y,y)/ \
            _np.einsum('...mnop,...m,...n,...o,...p',S_,x,x,x,x)


def K_V(*,C=None,S=None):
    """
    Equivalent isotropic bulk modulus (isostrain/Voigt assumption).

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    W. Voigt, Abhandlungen der Königlichen Gesellschaft der Wissenschaften in Göttingen: Mathematische Classe 34:3-52, 1887
    https://gdz.sub.uni-goettingen.de/id/PPN250442582_0034

    R. Hill, Proceedings of the Physical Society. Section A 66:349-354, 1952
    https://doi.org/10.1088/0370-1298/65/5/307
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if C is None: C = _np.linalg.inv(S)
    return ((C[...,0,0]+C[...,1,1]+C[...,2,2]) +2.*(C[...,0,1]+C[...,1,2]+C[...,2,0]))/9.

def K_R(*,C=None,S=None):
    """
    Equivalent isotropic bulk modulus (isostress/Reuss assumption).

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    A. Reuss, Zeitschrift für Angewandte Mathematik und Mechanik 9(1):49-53, 1929
    https://doi.org/10.1002/zamm.19290090104

    R. Hill, Proceedings of the Physical Society. Section A 66:349-354, 1952
    https://doi.org/10.1088/0370-1298/65/5/307
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)

    return 1./(S[...,0,0]+S[...,1,1]+S[...,2,2] +2.*(S[...,0,1]+S[...,1,2]+S[...,2,0]))


def G_V(*,C=None,S=None):
    """
    Equivalent isotropic shear modulus (isostrain/Voigt assumption).

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    W. Voigt, Abhandlungen der Königlichen Gesellschaft der Wissenschaften in Göttingen: Mathematische Classe 34:3-52, 1887
    https://gdz.sub.uni-goettingen.de/id/PPN250442582_0034

    R. Hill, Proceedings of the Physical Society. Section A 66:349-354, 1952
    https://doi.org/10.1088/0370-1298/65/5/307
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if C is None: C = _np.linalg.inv(S)

    return (     C[...,0,0]+C[...,1,1]+C[...,2,2]
               -(C[...,0,1]+C[...,1,2]+C[...,2,0])
            +3.*(C[...,3,3]+C[...,4,4]+C[...,5,5]))/15.

def G_R(*,C=None,S=None):
    """
    Equivalent isotropic shear modulus (isostress/Reuss assumption).

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    A. Reuss, Zeitschrift für Angewandte Mathematik und Mechanik 9(1):49-53, 1929
    https://doi.org/10.1002/zamm.19290090104

    R. Hill, Proceedings of the Physical Society. Section A 66:349-354, 1952
    https://doi.org/10.1088/0370-1298/65/5/307
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)

    return 15./( 4.*(S[...,0,0]+S[...,1,1]+S[...,2,2])
                -4.*(S[...,0,1]+S[...,1,2]+S[...,2,0])
                +3.*(S[...,3,3]+S[...,4,4]+S[...,5,5]))


def A_u(*,C=None,S=None):
    """
    Universal anisotropy index

    Parameters
    ----------
    C: numpy.ndarray, shape (...,6,6), optional
       Stiffness matrix in Voigt notation.
    S: numpy.ndarray, shape (...,6,6), optional
       Compliance matrix in Voigt notation.

    References
    ----------
    S. I. Ranganathan and M. Ostoja-Starzewski, Physical Review Letters 101:055504, 2008
    https://doi.org/10.1103/PhysRevLett.101.055504
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    C_ = C if C is not None else _np.linalg.inv(S)
    S_ = S if S is not None else _np.linalg.inv(C)
    return 5.*G_V(C=C_)/G_R(S=S_) + K_V(C=C_)/K_R(S=S_) - 6.
