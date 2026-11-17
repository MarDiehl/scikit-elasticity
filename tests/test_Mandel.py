# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from skel import Mandel


def test_basis_orthonormal(assert_allclose):
    assert_allclose(np.einsum('aij,bij->ab',Mandel.BASIS,Mandel.BASIS),np.eye(6))


def test_normal_components(assert_allclose):
    t = np.diag([1.0,2.0,3.0])
    assert_allclose(Mandel.to_6(t),np.array([1.0,2.0,3.0,0.0,0.0,0.0]))


def test_shear_components(assert_allclose):
    t = np.array([[0.0,1.0,2.0],
                  [1.0,0.0,3.0],
                  [2.0,3.0,0.0]])
    assert_allclose(Mandel.to_6(t),np.sqrt(2.0)*np.array([0.0,0.0,0.0,3.0,2.0,1.0]))


def test_stress_and_strain_identical(assert_allclose):
    t = np.array([[1.0,2.0,3.0],
                  [2.0,4.0,5.0],
                  [3.0,5.0,6.0]])
    assert_allclose(Mandel.to_6_stress(t),Mandel.to_6_strain(t))


def test_identity_stiffness(assert_allclose):
    I = 0.5*(np.einsum('ik,jl->ijkl',np.eye(3),np.eye(3))
            +np.einsum('il,jk->ijkl',np.eye(3),np.eye(3)))
    assert_allclose(Mandel.to_6x6(I),np.eye(6))
