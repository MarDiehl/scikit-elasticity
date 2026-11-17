# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from skel import Voigt


def test_basis_dual(assert_allclose):
    assert_allclose(np.einsum('aij,bij->ab',Voigt.BASIS_STRESS,Voigt.BASIS_STRAIN),np.eye(6))


def test_normal_components(assert_allclose):
    t = np.diag([1.0,2.0,3.0])
    expected = np.array([1.0,2.0,3.0,0.0,0.0,0.0])
    assert_allclose(Voigt.to_6_stress(t),expected)
    assert_allclose(Voigt.to_6_strain(t),expected)


def test_shear_components(assert_allclose):
    t = np.array([[0.0,1.0,2.0],
                  [1.0,0.0,3.0],
                  [2.0,3.0,0.0]])
    assert_allclose(Voigt.to_6_stress(t),np.array([0.0,0.0,0.0,3.0,2.0,1.0]))
    assert_allclose(Voigt.to_6_strain(t),np.array([0.0,0.0,0.0,6.0,4.0,2.0]))


def test_inverse_shear_components(assert_allclose):
    vector = np.array([0.0,0.0,0.0,1.0,0.0,0.0])
    assert_allclose(Voigt.to_3x3_stress(vector),
                    np.array([[0.0,0.0,0.0],
                              [0.0,0.0,1.0],
                              [0.0,1.0,0.0]]))
    assert_allclose(Voigt.to_3x3_strain(vector),
                    np.array([[0.0,0.0,0.0],
                              [0.0,0.0,0.5],
                              [0.0,0.5,0.0]]))
