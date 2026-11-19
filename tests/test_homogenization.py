# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import damask
import pytest
from scipy import linalg

import skel

METHODS = ['arithmetic_stiffness','arithmetic_compliance',
           'geometric_log','geometric_affine','super_hill']

def _to_Mandel(C_Voigt):
    """Convert a Voigt stiffness or compliance matrix to Mandel notation."""
    return skel.Mandel.to_6x6(skel.Voigt.to_3x3x3x3_stiffness(C_Voigt))

def _identity_rotation():
    """One-dimensional identity rotation."""
    return damask.Rotation.from_quaternion([[1.,0.,0.,0.]])

def test_equivalent(assert_allclose):
    C_11 = 124.1e9
    C_12 = 93.7e9
    C_44 = 46.4e9
    C = _to_Mandel(skel.Hooke.C('cubic',C_11=C_11,C_12=C_12,C_44=C_44))
    S = np.linalg.inv(C)
    r = _identity_rotation()

    for name in ['C_Reuss','C_Voigt']:
        function = getattr(skel.homogenization,name)
        assert_allclose(C,function(r,C=C))
        assert_allclose(C,function(r,S=S))

    for name in ['S_Reuss','S_Voigt']:
        function = getattr(skel.homogenization,name)
        assert_allclose(S,function(r,C=C))
        assert_allclose(S,function(r,S=S))

    assert_allclose(C,skel.homogenization.C_geo(r,C=C))
    assert_allclose(C,skel.homogenization.C_geo(r,S=S))
    assert_allclose(S,skel.homogenization.S_geo(r,C=C))
    assert_allclose(S,skel.homogenization.S_geo(r,S=S))

    for method in METHODS:
        assert_allclose(C,skel.homogenization.C_Hill(r,C=C,method=method))
        assert_allclose(C,skel.homogenization.C_Hill(r,S=S,method=method))
        assert_allclose(S,skel.homogenization.S_Hill(r,C=C,method=method))
        assert_allclose(S,skel.homogenization.S_Hill(r,S=S,method=method))
        assert_allclose(C@skel.homogenization.S_Hill(r,C=C,method=method),np.eye(6))

def test_input_equivalence(assert_allclose):
    """Providing C or S gives the same result for a nontrivial orientation set."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6,6))
    C = A@A.T + 6.*np.eye(6)
    C = (C+C.T)*.5
    S = np.linalg.inv(C)
    r = damask.Rotation.from_random(7,rng)

    for name in ['C_Voigt','C_Reuss','C_geo','C_sc']:
        function = getattr(skel.homogenization,name)
        assert_allclose(function(r,C=C),function(r,S=S))

    for name in ['S_Voigt','S_Reuss','S_geo','S_sc']:
        function = getattr(skel.homogenization,name)
        assert_allclose(function(r,C=C),function(r,S=S))

    for method in METHODS:
        assert_allclose(skel.homogenization.C_Hill(r,C=C,method=method),
                        skel.homogenization.C_Hill(r,S=S,method=method))
        assert_allclose(skel.homogenization.S_Hill(r,C=C,method=method),
                        skel.homogenization.S_Hill(r,S=S,method=method))

def test_hill_methods(assert_allclose):
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6,6))
    C = A@A.T + 6.*np.eye(6)
    C = (C+C.T)*.5
    r = damask.Rotation.from_random(7,rng)

    C_V = skel.homogenization.C_Voigt(r,C=C)
    C_R = skel.homogenization.C_Reuss(r,C=C)
    S_V = skel.homogenization.S_Voigt(r,C=C)
    S_R = skel.homogenization.S_Reuss(r,C=C)

    C_Hill_C = (C_V+C_R)*.5
    C_Hill_S = np.linalg.inv((S_V+S_R)*.5)
    C_log = np.real_if_close(
        linalg.expm((linalg.logm(C_V)+linalg.logm(C_R))*.5)
    )
    C_sqrt = linalg.sqrtm(C_V)
    C_inv_sqrt = np.linalg.inv(C_sqrt)
    C_affine = np.real_if_close(
        C_sqrt@linalg.sqrtm(C_inv_sqrt@C_R@C_inv_sqrt)@C_sqrt
    )

    assert_allclose(C_Hill_C,skel.homogenization.C_Hill(r,C=C,method='arithmetic_stiffness'))
    assert_allclose(C_Hill_S,skel.homogenization.C_Hill(r,C=C,method='arithmetic_compliance'))
    assert_allclose(C_log,skel.homogenization.C_Hill(r,C=C,method='geometric_log'))
    assert_allclose(C_affine,skel.homogenization.C_Hill(r,C=C,method='geometric_affine'))

    C_super = skel.homogenization.C_Hill(r,C=C,method='super_hill')
    S_super = skel.homogenization.S_Hill(r,C=C,method='super_hill')
    assert_allclose(C_super@S_super,np.eye(6))
    assert_allclose(C_super,C_super.T)
    assert_allclose(C_super,skel.homogenization.C_Hill(r,C=C))
    assert_allclose(S_super,skel.homogenization.S_Hill(r,C=C))

    for method in METHODS:
        C_Hill = skel.homogenization.C_Hill(r,C=C,method=method)
        S_Hill = skel.homogenization.S_Hill(r,C=C,method=method)
        assert_allclose(S_Hill,np.linalg.inv(C_Hill))

    assert not np.allclose(C_Hill_C,C_Hill_S)

def test_geo(assert_allclose):
    rng = np.random.default_rng(4)
    A = rng.normal(size=(6,6))
    C = A@A.T + 6.*np.eye(6)
    C = (C+C.T)*.5
    S = np.linalg.inv(C)
    r = damask.Rotation.from_random(7,rng)

    C_rotated = skel.Mandel.rotate_stiffness(r,C)
    C_ref = linalg.expm(np.average([linalg.logm(_) for _ in C_rotated],axis=0))

    C_geo = skel.homogenization.C_geo(r,C=C)
    S_geo = skel.homogenization.S_geo(r,C=C)

    assert_allclose(C_ref,C_geo)
    assert_allclose(C_geo,skel.homogenization.C_geo(r,S=S))
    assert_allclose(S_geo,skel.homogenization.S_geo(r,S=S))
    assert_allclose(S_geo,np.linalg.inv(C_geo))
    assert_allclose(C_geo@S_geo,np.eye(6))
    assert_allclose(C_geo,C_geo.T)
    assert (np.linalg.eigvalsh(C_geo) > 0.).all()

def test_invalid_input():
    C = np.eye(6)
    r = _identity_rotation()
    with pytest.raises(KeyError):
        skel.homogenization.C_Voigt(r)
    with pytest.raises(KeyError):
        skel.homogenization.C_Voigt(r,C=C,S=C)
    with pytest.raises(KeyError):
        skel.homogenization.C_Hill(r,C=C,method='invalid')

def test_rotate_only_stiffness(monkeypatch):
    """Stiffness-only schemes do not rotate compliance."""
    C = np.eye(6)
    r = _identity_rotation()

    def fail(*args,**kwargs):
        raise AssertionError('unneeded rotation performed')

    monkeypatch.setattr(skel.Mandel,'rotate_compliance',fail)
    skel.homogenization.C_Voigt(r,C=C)
    skel.homogenization.S_Voigt(r,C=C)

def test_rotate_only_compliance(monkeypatch):
    """Compliance-only schemes do not rotate stiffness."""
    S = np.eye(6)
    r = _identity_rotation()

    def fail(*args,**kwargs):
        raise AssertionError('unneeded rotation performed')

    monkeypatch.setattr(skel.Mandel,'rotate_stiffness',fail)
    skel.homogenization.C_Reuss(r,S=S)
    skel.homogenization.S_Reuss(r,S=S)

def test_geo_rotate_only_stiffness(monkeypatch):
    """The geometric mean does not rotate compliance."""
    C = np.eye(6)
    r = _identity_rotation()

    def fail(*args,**kwargs):
        raise AssertionError('unneeded rotation performed')

    monkeypatch.setattr(skel.Mandel,'rotate_compliance',fail)
    skel.homogenization.C_geo(r,C=C)
    skel.homogenization.S_geo(r,C=C)

def test_geometric_means(assert_allclose):
    """Both geometric methods are symmetric and inverse-consistent."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6,6))
    C = A@A.T + 6.*np.eye(6)
    C = (C+C.T)*.5
    S = np.linalg.inv(C)
    r = damask.Rotation.from_random(7,rng)

    for method in ['geometric_log','geometric_affine']:
        C_geo = skel.homogenization.C_Hill(r,C=C,method=method)
        S_geo = skel.homogenization.S_Hill(r,C=C,method=method)

        assert_allclose(C_geo,skel.homogenization.C_Hill(r,S=S,method=method))
        assert_allclose(S_geo,skel.homogenization.S_Hill(r,S=S,method=method))

        assert_allclose(C_geo,C_geo.T)
        assert_allclose(S_geo,S_geo.T)
        assert_allclose(C_geo@S_geo,np.eye(6))
        assert (np.linalg.eigvalsh(C_geo) > 0.).all()

    assert not np.allclose(
        skel.homogenization.C_Hill(r,C=C,method='geometric_log'),
        skel.homogenization.C_Hill(r,C=C,method='geometric_affine')
    )

def test_self_consistent_single_crystal(assert_allclose):
    """Self-consistent homogenization of a single crystal returns its stiffness."""
    C = _to_Mandel(skel.Hooke.C('cubic',C_11=124.1e9,C_12=93.7e9,C_44=46.4e9))
    S = np.linalg.inv(C)
    r = _identity_rotation()

    assert_allclose(skel.homogenization.C_sc(r,C=C),skel.Mandel.rotate_stiffness(r,C)[0])
    assert_allclose(skel.homogenization.C_sc(r,S=S),skel.Mandel.rotate_stiffness(r,C)[0])
    assert_allclose(skel.homogenization.S_sc(r,C=C),skel.Mandel.rotate_compliance(r,S)[0])
    assert_allclose(skel.homogenization.S_sc(r,S=S),skel.Mandel.rotate_compliance(r,S)[0])
