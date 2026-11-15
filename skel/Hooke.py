# SPDX-License-Identifier: AGPL-3.0-or-later
# https://dictionary.iucr.org/Crystal_system
# https://dictionary.iucr.org/Lattice_system
# https://dictionary.iucr.org/Crystal_family
r"""
Linear elasticity according to the generalized Hooke's law.

This module assembles stiffness and compliance matrices in Voigt
notation and provides directional and isotropic elastic properties.

Symmetry
--------
The form of the elastic tensor is determined by the point group of the
crystal. Elasticity is a centrosymmetric fourth-rank tensor, so the
relevant symmetry is the Laue class, i.e. the point group including
inversion.

The ``crystal_system`` argument is a user-facing shorthand for the most
general Laue-class form of a crystal system. The higher-symmetry class
is obtained by omitting the optional constants or setting them to zero.

The relation between crystal systems, Laue classes, and the independent
constants used here is:

``triclinic``
    Laue class :math:`\bar{1}`; 21 independent constants.
``monoclinic``
    Laue class :math:`2/m`; 13 independent constants. The twofold axis
    is parallel to :math:`x_2` (Nye's standard orientation).
``orthorhombic``
    Laue class :math:`mmm`; 9 independent constants.
``tetragonal``
    Laue classes :math:`4/m` and :math:`4/mmm`. The general form has 7
    constants with :math:`C_{16}=-C_{26}`. For :math:`4/mmm`, set
    :math:`C_{16}=C_{26}=0`, giving 6 constants.
``trigonal``
    Laue classes :math:`\bar{3}` and :math:`\bar{3}m`. The general form
    has 7 constants with :math:`C_{14}=-C_{24}=C_{56}` and
    :math:`C_{15}=-C_{25}=-C_{46}`. For :math:`\bar{3}m`, set
    :math:`C_{15}=C_{25}=C_{46}=0`, giving 6 constants.
``hexagonal``
    Laue classes :math:`6/m` and :math:`6/mmm`; 5 independent constants
    with :math:`C_{66}=(C_{11}-C_{12})/2`.
``cubic``
    Laue classes :math:`m\bar{3}` and :math:`m\bar{3}m`; 3 independent
    constants.
``isotropic``
    Not a crystal system; 2 independent constants with
    :math:`C_{44}=(C_{11}-C_{12})/2`.

All entries not listed as independent are either zero or related by the
corresponding symmetry.
"""

import numpy as _np
from . import Voigt as _Voigt

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
    crystal_system : str
        Crystal system.
    mode : {'C', 'S'}
        Assemble a stiffness or compliance matrix.
    X_11, X_12, ..., X_66 : float
        Upper-triangle entries.

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
      - X_14 = -X_24 = X_56/d
      - X_15 = -X_25 = -X_46/d (or 0.0)
    tetragonal:
      - X_11 = X_22
      - X_13 = X_23
      - X_44 = X_55
      - X_16 = -X_26 (or 0.0)

    f = 0.5 for C, 2.0 for S
    d = 1.0 for C, 2.0 for S

    References
    ----------
    J. F. Nye, *Physical Properties of Crystals*, Oxford University Press,
    1985, Table 9, p. 140.
    """
    match mode:
        case 'C':
            f, d = 0.5, 1.0
        case 'S':
            f, d = 2.0, 2.0
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
                X_11=X_22=X_33 = X_44/f + X_12
            elif X_12 is None and X_11 is not None and X_44 is not None:
                X_12=X_13=X_23 = X_11 - X_44/f
            elif X_44 is None and X_11 is not None and X_12 is not None:
                X_44=X_55=X_66 = f*(X_11-X_12)
            elif not _np.isclose(X_44,f*(X_11-X_12)):
                raise ValueError(f'condition {mode}_44 = {f}×({mode}_11 - {mode}_12) for isotropic crystal system violated')


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
                X_11=X_22 = X_66/f + X_12
            elif X_12 is None and [X_11,X_66].count(None) == 0:
                X_12 = X_11 - X_66/f
            elif X_66 is None and [X_11,X_12].count(None) == 0:
                X_66 = f*(X_11-X_12)
            elif not _np.isclose(X_66,f*(X_11-X_12)):
                raise ValueError(f'condition {mode}_66 = {f}×({mode}_11 - {mode}_12) for hexagonal crystal system violated')


        case 'trigonal':
            d_ = '' if d == 1.0 else f'/{d:g}'
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
                X_11=X_22 = X_66/f + X_12
            elif X_12 is None and [X_11,X_66].count(None) == 0:
                X_12 = X_11 - X_66/f
            elif X_66 is None and [X_11,X_12].count(None) == 0:
                X_66 = f*(X_11-X_12)
            elif not _np.isclose(X_66,f*(X_11-X_12)):
                raise ValueError(f'condition {mode}_66 = {f}×({mode}_11 - {mode}_12) for trigonal crystal system violated')

            msg = f'condition {mode}_14 = -{mode}_24 = {mode}_56{d_} for trigonal crystal system violated'
            match [X_14,X_24,X_56].count(None):
                case 3:
                    raise ValueError(msg)
                case 2:
                    if X_14 is None and X_24 is None:
                        X_14 = X_56/d
                        X_24 = -X_56/d
                    elif X_14 is None and X_56 is None:
                        X_14 = -X_24
                        X_56 = -d*X_24
                    else:
                        X_24 = -X_14
                        X_56 = d*X_14
                case 1:
                    if X_14 is None:
                        X_14 = -X_24
                        if X_24 != -X_56/d: raise ValueError(msg)
                    elif X_24 is None:
                        X_24 = -X_14
                        if X_14 != X_56/d: raise ValueError(msg)
                    else:
                        X_56 = d*X_14
                        if X_14 != -X_24: raise ValueError(msg)
                case 0:
                    if X_14 != -X_24 or X_14 != X_56/d:
                        raise ValueError(msg)

            msg = f'condition {mode}_15 = -{mode}_25 = -{mode}_46{d_} for trigonal crystal system violated'
            match [X_15,X_25,X_46].count(None):
                case 3:
                    X_15=X_25=X_46 = 0.0
                case 2:
                    if X_15 is None and X_25 is None:
                        X_15 = -X_46/d
                        X_25 = X_46/d
                    elif X_15 is None and X_46 is None:
                        X_15 = -X_25
                        X_46 = d*X_25
                    else:
                        X_25 = -X_15
                        X_46 = -d*X_15
                case 1:
                    if X_15 is None:
                        X_15 = -X_25
                        if X_25 != X_46/d: raise ValueError(msg)
                    elif X_25 is None:
                        X_25 = -X_15
                        if X_15 != -X_46/d: raise ValueError(msg)
                    else:
                        X_46 = -d*X_15
                        if X_15 != -X_25: raise ValueError(msg)
                case 0:
                    if X_15 != -X_25 or X_15 != -X_46/d:
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
                    raise ValueError(f'condition {mode}_16 = -{mode}_26 for tetragonal crystal system violated')


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
                             X_34=X_34,          X_36=X_36,
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
    r"""
    Assemble a stiffness matrix.

    Parameters
    ----------
    crystal_system : {'isotropic', 'cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'}
        Crystal system. The assembled matrix uses the most general
        Laue-class form of this system; see the module documentation
        for the corresponding symmetry relations.
    C_11, C_12, ..., C_66 : float, optional
        Upper-triangle entries in Voigt notation. Only the independent
        entries for the selected crystal system need to be given.
        Dependent entries are inferred from the symmetry relations.
        If a dependent entry is supplied, it must be consistent with
        the inferred value.

    Returns
    -------
    C : numpy.ndarray of shape (6,6)
        Stiffness matrix in Voigt notation.

    Raises
    ------
    ValueError
        If the supplied entries violate the symmetry relations or the
        resulting matrix is not stable.

    Notes
    -----
    Only the upper triangle is given; the lower triangle is filled by
    symmetry. The following optional relations are used when entries are
    omitted:

    - tetragonal: :math:`C_{16}=-C_{26}`. If both are omitted, both are
      set to zero (:math:`4/mmm`).
    - trigonal: :math:`C_{14}=-C_{24}=C_{56}` and
      :math:`C_{15}=-C_{25}=-C_{46}`. At least one of
      :math:`C_{14},C_{24},C_{56}` must be given. If
      :math:`C_{15},C_{25},C_{46}` are all omitted, they are set to
      zero (:math:`\bar{3}m`).
    - hexagonal: :math:`C_{66}=(C_{11}-C_{12})/2`.
    - isotropic: :math:`C_{44}=(C_{11}-C_{12})/2`.

    The resulting matrix is checked with :func:`stable`.

    Examples
    --------
    >>> import skel
    >>> C_Fe = skel.Hooke.C('cubic',C_11=232.2e9,C_12=136.4e9,C_44=117.0e9)
    >>> C_Mg = skel.Hooke.C('hexagonal',C_11=59.5e9,C_12=25.6e9,
    ...                     C_44=16.5e9,C_33=61.7e9,C_13=21.5e9)
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
    r"""
    Assemble a compliance matrix.

    Parameters
    ----------
    crystal_system : {'isotropic', 'cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'}
        Crystal system. The assembled matrix uses the most general
        Laue-class form of this system; see the module documentation
        for the corresponding symmetry relations.
    S_11, S_12, ..., S_66 : float, optional
        Upper-triangle entries in Voigt notation. Only the independent
        entries for the selected crystal system need to be given.
        Dependent entries are inferred from the symmetry relations.
        If a dependent entry is supplied, it must be consistent with
        the inferred value.

    Returns
    -------
    S : numpy.ndarray of shape (6,6)
        Compliance matrix in Voigt notation.

    Raises
    ------
    ValueError
        If the supplied entries violate the symmetry relations or the
        resulting matrix is not stable.

    Notes
    -----
    Only the upper triangle is given; the lower triangle is filled by
    symmetry. The compliance relations differ from the stiffness
    relations because of the Voigt weights:

    - tetragonal: :math:`S_{16}=-S_{26}`. If both are omitted, both are
      set to zero (:math:`4/mmm`).
    - trigonal: :math:`S_{14}=-S_{24}=S_{56}/2` and
      :math:`S_{15}=-S_{25}=-S_{46}/2`. At least one of
      :math:`S_{14},S_{24},S_{56}` must be given. If
      :math:`S_{15},S_{25},S_{46}` are all omitted, they are set to
      zero (:math:`\bar{3}m`).
    - hexagonal: :math:`S_{66}=2(S_{11}-S_{12})`.
    - isotropic: :math:`S_{44}=2(S_{11}-S_{12})`.

    The resulting matrix is checked with :func:`stable`.

    Examples
    --------
    >>> import numpy as np
    >>> import skel
    >>> C_Fe = skel.Hooke.C('cubic',C_11=232.2e9,C_12=136.4e9,C_44=117.0e9)
    >>> S_Fe = np.linalg.inv(C_Fe)
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
    r"""
    Determine whether a stiffness or compliance matrix is stable.

    Parameters
    ----------
    crystal_system : {'isotropic', 'cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'}
        Crystal system of the matrix. The stability criteria depend on
        the corresponding Laue class.
    C : numpy.ndarray of shape (6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (6,6), optional
        Compliance matrix in Voigt notation.

    Returns
    -------
    bool
        ``True`` if the matrix satisfies the Born stability criteria.

    Raises
    ------
    TypeError
        If neither or both of ``C`` and ``S`` are given.
    ValueError
        If ``crystal_system`` is not a valid crystal system.

    Notes
    -----
    The criteria are necessary and sufficient for elastic stability in
    the harmonic approximation and in the absence of external stress.
    They are taken from Mouhat and Coudert, who give them per Laue
    class.

    For tetragonal and trigonal systems, the implementation uses the
    more general :math:`4/m` and :math:`\bar{3}` conditions. The
    higher-symmetry :math:`4/mmm` and :math:`\bar{3}m` cases are
    obtained by setting :math:`C_{16}=C_{26}=0` and
    :math:`C_{15}=C_{25}=C_{46}=0`, respectively.

    The matrix must have the symmetry of the selected crystal system.
    These criteria are not a general positive-definiteness test for an
    arbitrary 6x6 matrix.

    If ``S`` is given, it is inverted to obtain ``C`` for the
    stability check.

    References
    ----------
    F. Mouhat and F.-X. Coudert, Physical Review B 90:224104, 2014.
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
            # Mouhat and Coudert call this rhombohedral; Nye uses trigonal.
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
            raise ValueError(f'invalid crystal system {crystal_system}')


def _prepare_directions(*vectors,normalize=False,orthogonal=False):
    """
    Validate and optionally normalize direction vectors.

    Parameters
    ----------
    *vectors : numpy.ndarray of shape (...,3)
        Direction vectors.
    normalize : bool, optional
        Normalize the vectors to unit length before checking. Defaults
        to False.
    orthogonal : bool, optional
        Require the first two vectors to be orthogonal. Defaults to
        False.

    Returns
    -------
    tuple of numpy.ndarray
        Prepared vectors.
    """
    vectors = tuple(_np.asarray(v) for v in vectors)
    if normalize:
        norms = tuple(_np.linalg.norm(v,axis=-1,keepdims=True) for v in vectors)
        if any(_np.any(norm == 0.0) for norm in norms):
            raise ValueError('direction vectors must not be zero')
        vectors = tuple(v/norm for v,norm in zip(vectors,norms))
    for v in vectors:
        if not _np.all(_np.isclose(_np.linalg.norm(v,axis=-1),1.0,rtol=1e-5,atol=1e-8)):
            raise ValueError('direction vectors must have unit length; use normalize=True')
    if orthogonal:
        if not _np.all(_np.isclose(_np.einsum('...i,...i',vectors[0],vectors[1]),0.0,atol=1e-8)):
            raise ValueError('direction vectors must be orthogonal')
    return vectors


def E(x,*,C=None,S=None,normalize=False):
    r"""
    Directional Young's modulus.

    Parameters
    ----------
    x : numpy.ndarray of shape (...,3)
        Unit vector(s) along the loading direction.
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.
    normalize : bool, optional
        Normalize ``x`` to unit length before evaluation. Defaults to
        False, in which case non-unit vectors raise ``ValueError``.

    Returns
    -------
    E : numpy.ndarray
        Young's modulus along ``x``.

    Notes
    -----
    .. math::

        E(\mathbf{x}) = \frac{1}{S_{ijkl} x_i x_j x_k x_l}

    References
    ----------
    https://mtex-toolbox.github.io/stiffnessTensor.YoungsModulus.html
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)
    x, = _prepare_directions(x,normalize=normalize)

    return 1/_np.einsum('...ijkl,...i,...j,...k,...l',_Voigt.to_3x3x3x3_compliance(S),x,x,x,x)

def G(h,u,*,C=None,S=None,normalize=False):
    r"""
    Directional shear modulus.

    Parameters
    ----------
    h : numpy.ndarray of shape (...,3)
        Unit vector(s) normal to the shear plane.
    u : numpy.ndarray of shape (...,3)
        Unit vector(s) in the shear direction. Must be orthogonal to
        ``h``.
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.
    normalize : bool, optional
        Normalize ``h`` and ``u`` to unit length before evaluation.
        Defaults to False, in which case non-unit vectors raise
        ``ValueError``. Orthogonality is always checked.

    Returns
    -------
    G : numpy.ndarray
        Shear modulus for the plane/direction pair ``(h, u)``.

    Notes
    -----
    .. math::

        G(\mathbf{h},\mathbf{u}) =
        \frac{1}{4 S_{ijkl} h_i u_j h_k u_l}

    References
    ----------
    https://mtex-toolbox.github.io/stiffnessTensor.shearModulus.html
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)
    h,u = _prepare_directions(h,u,normalize=normalize,orthogonal=True)

    return .25/_np.einsum('...ijkl,...i,...j,...k,...l',_Voigt.to_3x3x3x3_compliance(S),h,u,h,u)

def nu(x,y,*,C=None,S=None,normalize=False):
    r"""
    Directional Poisson ratio.

    Parameters
    ----------
    x : numpy.ndarray of shape (...,3)
        Unit vector(s) along the loading direction.
    y : numpy.ndarray of shape (...,3)
        Unit vector(s) along the transverse direction. Must be
        orthogonal to ``x``.
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.
    normalize : bool, optional
        Normalize ``x`` and ``y`` to unit length before evaluation.
        Defaults to False, in which case non-unit vectors raise
        ``ValueError``. Orthogonality is always checked.

    Returns
    -------
    nu : numpy.ndarray
        Poisson ratio for loading along ``x`` and transverse direction
        ``y``.

    Notes
    -----
    .. math::

        \nu(\mathbf{x},\mathbf{y}) =
        -\frac{S_{ijkl} x_i x_j y_k y_l}
               {S_{mnop} x_m x_n x_o x_p}

    References
    ----------
    https://mtex-toolbox.github.io/stiffnessTensor.PoissonRatio.html
    """
    if sum(arg is not None for arg in (C,S)) != 1:
        raise TypeError('missing keyord argument "C" or "S"')
    if S is None: S = _np.linalg.inv(C)
    x,y = _prepare_directions(x,y,normalize=normalize,orthogonal=True)

    S_ = _Voigt.to_3x3x3x3_compliance(S)
    return -_np.einsum('...ijkl,...i,...j,...k,...l',S_,x,x,y,y)/ \
            _np.einsum('...mnop,...m,...n,...o,...p',S_,x,x,x,x)


def K_V(*,C=None,S=None):
    r"""
    Equivalent isotropic bulk modulus (isostrain/Voigt assumption).

    Parameters
    ----------
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.

    Returns
    -------
    K_V : numpy.ndarray
        Voigt bulk modulus.

    Notes
    -----
    .. math::

        K_V = \frac{C_{11}+C_{22}+C_{33}
                     +2(C_{12}+C_{13}+C_{23})}{9}

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
    r"""
    Equivalent isotropic bulk modulus (isostress/Reuss assumption).

    Parameters
    ----------
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.

    Returns
    -------
    K_R : numpy.ndarray
        Reuss bulk modulus.

    Notes
    -----
    .. math::

        K_R = \frac{1}{S_{11}+S_{22}+S_{33}
                          +2(S_{12}+S_{13}+S_{23})}

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
    r"""
    Equivalent isotropic shear modulus (isostrain/Voigt assumption).

    Parameters
    ----------
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.

    Returns
    -------
    G_V : numpy.ndarray
        Voigt shear modulus.

    Notes
    -----
    .. math::

        G_V = \frac{C_{11}+C_{22}+C_{33}
                     -(C_{12}+C_{13}+C_{23})
                     +3(C_{44}+C_{55}+C_{66})}{15}

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
    r"""
    Equivalent isotropic shear modulus (isostress/Reuss assumption).

    Parameters
    ----------
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.

    Returns
    -------
    G_R : numpy.ndarray
        Reuss shear modulus.

    Notes
    -----
    .. math::

        G_R = \frac{15}{4(S_{11}+S_{22}+S_{33})
                        -4(S_{12}+S_{13}+S_{23})
                        +3(S_{44}+S_{55}+S_{66})}

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
    r"""
    Universal anisotropy index.

    Parameters
    ----------
    C : numpy.ndarray of shape (...,6,6), optional
        Stiffness matrix in Voigt notation.
    S : numpy.ndarray of shape (...,6,6), optional
        Compliance matrix in Voigt notation.

    Returns
    -------
    A_u : numpy.ndarray
        Universal anisotropy index.

    Notes
    -----
    .. math::

        A^U = 5\frac{G_V}{G_R} + \frac{K_V}{K_R} - 6

    The index is zero for isotropic elasticity and positive otherwise.
    It is defined for any elastic tensor and therefore requires no
    crystal-system argument.

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
