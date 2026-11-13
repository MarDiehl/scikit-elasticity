# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from skel import _conventions
from skel import Voigt
from skel import Mandel
from skel import natural


@pytest.mark.parametrize('forward,backward',[(Voigt.to_3x3_strain, Voigt.to_6_strain),
                                             (Voigt.to_3x3_stress, Voigt.to_6_stress),
                                             (Mandel.to_3x3_strain, Mandel.to_6_strain),
                                             (Mandel.to_3x3_stress, Mandel.to_6_stress),
                                             (natural.to_3x3_strain, natural.to_6_strain),
                                             (natural.to_3x3_stress, natural.to_6_stress)])
@pytest.mark.parametrize('ndim',[0,1,3])
def test_conversion_stress_strain(random_6,random_shape,forward,backward,ndim,assert_allclose):
    t = random_6(random_shape(ndim))
    assert_allclose(backward(forward(t)),t)


@pytest.mark.parametrize('forward,backward',[(Voigt.to_3x3x3x3_stiffness, Voigt.to_6x6_stiffness),
                                             (Voigt.to_3x3x3x3_compliance, Voigt.to_6x6_compliance),
                                             (Mandel.to_3x3x3x3_stiffness, Mandel.to_6x6_stiffness),
                                             (Mandel.to_3x3x3x3_compliance, Mandel.to_6x6_compliance),
                                             (natural.to_3x3x3x3_stiffness, natural.to_6x6_stiffness),
                                             (natural.to_3x3x3x3_compliance, natural.to_6x6_compliance)])
@pytest.mark.parametrize('ndim',[0,1,3])
def test_conversion_stiffness_compliance(symmetric_6x6,random_shape,forward,backward,ndim,assert_allclose):
    t = symmetric_6x6(random_shape(ndim))
    assert_allclose(backward(forward(t)),t)



@pytest.mark.parametrize('to_3x3',[Voigt.to_3x3_strain,
                                   Voigt.to_3x3_stress,
                                   Mandel.to_3x3_strain,
                                   Mandel.to_3x3_stress,
                                   natural.to_3x3_strain,
                                   natural.to_3x3_stress])
def test_squeeze_6_to_3x3(to_3x3,random_6,assert_allclose):
    t = random_6()
    assert_allclose(to_3x3(t),to_3x3(t.reshape(1,1,1,6)).squeeze())


@pytest.mark.parametrize('to_6',[Voigt.to_6_strain,
                                 Voigt.to_6_stress,
                                 Mandel.to_6_strain,
                                 Mandel.to_6_stress,
                                 natural.to_6_strain,
                                 natural.to_6_stress])
def test_squeeze_3x3_to_6(to_6,symmetric_2nd,assert_allclose):
    t = symmetric_2nd()
    assert_allclose(to_6(t),to_6(t.reshape(1,1,1,3,3)).squeeze())


@pytest.mark.parametrize('to_3x3x3x3',[Voigt.to_3x3x3x3_compliance,
                                       Voigt.to_3x3x3x3_stiffness,
                                       Mandel.to_3x3x3x3_compliance,
                                       Mandel.to_3x3x3x3_stiffness,
                                       natural.to_3x3x3x3_compliance,
                                       natural.to_3x3x3x3_stiffness])
def test_squeeze_6x6_to_3x3x3x3(to_3x3x3x3,symmetric_6x6,assert_allclose):
    t = symmetric_6x6()
    assert_allclose(to_3x3x3x3(t),to_3x3x3x3(t.reshape(1,1,1,6,6)).squeeze())


@pytest.mark.parametrize('to_6x6',[Voigt.to_6x6_compliance,
                                   Voigt.to_6x6_stiffness,
                                   Mandel.to_6x6_compliance,
                                   Mandel.to_6x6_stiffness,
                                   natural.to_6x6_compliance,
                                   natural.to_6x6_stiffness])
def test_squeeze_3x3x3x3_to_6x6(to_6x6,symmetric_6x6,assert_allclose):
    t = _conventions.convert_6x6_to_3x3x3x3(symmetric_6x6(),[1,1])
    assert_allclose(to_6x6(t),to_6x6(t.reshape(1,1,1,3,3,3,3)).squeeze())



@pytest.mark.parametrize('forward,backward,rotate',
                         [(Voigt.to_6_strain,Voigt.to_3x3_strain,Voigt.rotate_strain),
                          (Voigt.to_6_stress,Voigt.to_3x3_stress,Voigt.rotate_stress),
                          (Mandel.to_6_strain,Mandel.to_3x3_strain,Mandel.rotate_strain),
                          (Mandel.to_6_stress,Mandel.to_3x3_stress,Mandel.rotate_stress),
                          (natural.to_6_strain, natural.to_3x3_strain, natural.rotate_strain),
                          (natural.to_6_stress, natural.to_3x3_stress, natural.rotate_stress)])
@pytest.mark.parametrize('ndim',[0,1,3])
def test_rotation_2nd(symmetric_2nd,random_rotation,random_shape,forward,backward,rotate,ndim,assert_allclose):
    shape = random_shape(ndim)
    t = symmetric_2nd(shape)
    O = random_rotation(shape)
    t_6 = forward(t)
    t_6_rotated = rotate(O,t_6)
    assert_allclose(O@t,backward(t_6_rotated))

@pytest.mark.parametrize('forward,backward,rotate',
                         [(Voigt.to_6x6_compliance,Voigt.to_3x3x3x3_compliance,Voigt.rotate_compliance),
                          (Voigt.to_6x6_stiffness,Voigt.to_3x3x3x3_stiffness,Voigt.rotate_stiffness),
                          (Mandel.to_6x6_compliance,Mandel.to_3x3x3x3_compliance,Mandel.rotate_compliance),
                          (Mandel.to_6x6_stiffness,Mandel.to_3x3x3x3_stiffness,Mandel.rotate_stiffness),
                          (natural.to_6x6_compliance, natural.to_3x3x3x3_compliance, natural.rotate_compliance),
                          (natural.to_6x6_stiffness, natural.to_3x3x3x3_stiffness, natural.rotate_stiffness)])
@pytest.mark.parametrize('ndim',[0,1,3])
def test_rotation_4th(symmetric_6x6,random_rotation,random_shape,forward,backward,rotate,ndim,assert_allclose):
    shape = random_shape(ndim)
    t = _conventions.convert_6x6_to_3x3x3x3(symmetric_6x6(shape),[1,1])
    O = random_rotation(shape)
    t_6x6 = forward(t)
    t_6x6_rotated = rotate(O,t_6x6)
    assert_allclose(O@t,backward(t_6x6_rotated))

@pytest.mark.parametrize('stiffness_6x6_to_3x3x3x3,strain_6_to_3x3,stress_3x3_to_6',
                         [(Voigt.to_3x3x3x3_stiffness,Voigt.to_3x3_strain,Voigt.to_6_stress),
                          (Mandel.to_3x3x3x3_stiffness,Mandel.to_3x3_strain,Mandel.to_6_stress),
                          (natural.to_3x3x3x3_stiffness, natural.to_3x3_strain, natural.to_6_stress)])
@pytest.mark.parametrize('ndim',[0,1,3])
def test_stress_calculation(random_6,symmetric_6x6,random_shape,
                            stiffness_6x6_to_3x3x3x3,strain_6_to_3x3,stress_3x3_to_6,ndim,assert_allclose):
    shape = random_shape(ndim)
    epsilon_tilde = random_6(shape)
    C_tilde = symmetric_6x6(shape)

    C = stiffness_6x6_to_3x3x3x3(C_tilde)
    epsilon = strain_6_to_3x3(epsilon_tilde)

    sigma = np.einsum('...ijkl,...kl',C,epsilon)
    sigma_tilde = np.einsum('...ij,...j',C_tilde,epsilon_tilde)
    assert_allclose(stress_3x3_to_6(sigma),sigma_tilde)

@pytest.mark.parametrize('compliance_6x6_to_3x3x3x3,stress_6_to_3x3,strain_3x3_to_6',
                         [(Voigt.to_3x3x3x3_compliance,Voigt.to_3x3_stress,Voigt.to_6_strain),
                          (Mandel.to_3x3x3x3_compliance,Mandel.to_3x3_stress,Mandel.to_6_strain),
                          (natural.to_3x3x3x3_compliance, natural.to_3x3_stress, natural.to_6_strain)])
@pytest.mark.parametrize('ndim',[0,1,3])
def test_strain_calculation(random_6,symmetric_6x6,random_shape,
                            compliance_6x6_to_3x3x3x3,stress_6_to_3x3,strain_3x3_to_6,ndim,assert_allclose):
    shape = random_shape(ndim)
    sigma_tilde = random_6(shape)
    S_tilde = symmetric_6x6(shape)

    S = compliance_6x6_to_3x3x3x3(S_tilde)
    sigma = stress_6_to_3x3(sigma_tilde)

    epsilon = np.einsum('...ijkl,...kl',S,sigma)
    epsilon_tilde = np.einsum('...ij,...j',S_tilde,sigma_tilde)
    assert_allclose(strain_3x3_to_6(epsilon),epsilon_tilde)


@pytest.mark.parametrize('ndim_t',[0,1,3])
@pytest.mark.parametrize('ndim_O',[0,1,3])
def test_broadcast_2nd(random_6,random_rotation,random_shape,ndim_t,ndim_O):
    t_tilde = random_6(random_shape(ndim_t))
    O = random_rotation(random_shape(ndim_O))
    t_tilde_ = _conventions.rotate(O,t_tilde,[1.]*2,2)
    t_ = O@_conventions.convert_6_to_3x3(t_tilde,1.)
    assert t_tilde_.shape[:-1] == t_.shape[:-2]

@pytest.mark.parametrize('ndim_T',[0,1,3])
@pytest.mark.parametrize('ndim_O',[0,1,3])
def test_broadcast_4th(symmetric_6x6,random_rotation,random_shape,ndim_T,ndim_O):
    T_tilde = symmetric_6x6(random_shape(ndim_T))
    O = random_rotation(random_shape(ndim_O))
    T_tilde_ = _conventions.rotate(O,T_tilde,[1.]*2,4)
    T_ = O@_conventions.convert_6x6_to_3x3x3x3(T_tilde,[1.]*2)
    assert T_tilde_.shape[:-2] == T_.shape[:-4]
