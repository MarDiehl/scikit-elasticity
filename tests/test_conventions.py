# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest
import damask
from damask import tensor

from skel import _conventions
from skel import Voigt
from skel import Mandel


def t6():
    rng = np.random.default_rng()
    return [rng.random(6),
            rng.random(tuple(rng.integers(4,11,size=1))+(6,)),
            rng.random(tuple(rng.integers(4,11,size=3))+(6,))]
def t6x6():
    rng = np.random.default_rng()
    return [tensor.symmetric(rng.random((6,6))),
            tensor.symmetric(rng.random(tuple(rng.integers(4,11,size=1))+(6,6))),
            tensor.symmetric(rng.random(tuple(rng.integers(4,11,size=3))+(6,6)))]

def t6_t6x6():
    rng = np.random.default_rng()
    size1 = tuple(rng.integers(4,11,size=1))
    size3 = tuple(rng.integers(4,11,size=3))
    return [(rng.random(6),tensor.symmetric(rng.random((6,6)))),
            (rng.random(size1+(6,)),tensor.symmetric(rng.random(size1+(6,6)))),
            (rng.random(size3+(6,)),tensor.symmetric(rng.random(size3+(6,6))))]

def t3x3():
    rng = np.random.default_rng()
    return [tensor.symmetric(rng.random((3,3))),
            tensor.symmetric(rng.random(tuple(rng.integers(4,11,size=1))+(3,3))),
            tensor.symmetric(rng.random(tuple(rng.integers(4,11,size=3))+(3,3)))]

def O():
    rng = np.random.default_rng()
    return [damask.Rotation.from_random(),
            damask.Rotation.from_random(tuple(rng.integers(4,11,size=1))),
            damask.Rotation.from_random(tuple(rng.integers(4,11,size=3)))]


@pytest.mark.parametrize('forward,backward',[(Voigt.to_3x3_strain, Voigt.to_6_strain),
                                             (Voigt.to_3x3_stress, Voigt.to_6_stress),
                                             (Mandel.to_3x3_strain, Mandel.to_6_strain),
                                             (Mandel.to_3x3_stress, Mandel.to_6_stress)])
@pytest.mark.parametrize('t',t6())
def test_conversion_stress_strain(t,forward,backward):
    assert np.allclose(backward(forward(t)),t)


@pytest.mark.parametrize('forward,backward',[(Voigt.to_3x3x3x3_stiffness, Voigt.to_6x6_stiffness),
                                             (Voigt.to_3x3x3x3_compliance, Voigt.to_6x6_compliance),
                                             (Mandel.to_3x3x3x3_stiffness, Mandel.to_6x6_stiffness),
                                             (Mandel.to_3x3x3x3_compliance, Mandel.to_6x6_compliance)])
@pytest.mark.parametrize('t',t6x6())
def test_conversion_stiffness_compliance(t,forward,backward):
    assert np.allclose(backward(forward(t)),t)



@pytest.mark.parametrize('to_3x3',[Voigt.to_3x3_strain,
                                   Voigt.to_3x3_stress,
                                   Mandel.to_3x3_strain,
                                   Mandel.to_3x3_stress])
def test_squeeze_6_to_3x3(to_3x3):
    t = np.random.random(6)
    assert np.allclose(to_3x3(t),to_3x3(t.reshape(1,1,1,6)).squeeze())


@pytest.mark.parametrize('to_6',[Voigt.to_6_strain,
                                 Voigt.to_6_stress,
                                 Mandel.to_6_strain,
                                 Mandel.to_6_stress])
def test_squeeze_3x3_to_6(to_6):
    t = tensor.symmetric(np.random.random((3,3)))
    assert np.allclose(to_6(t),to_6(t.reshape(1,1,1,3,3)).squeeze())


@pytest.mark.parametrize('to_3x3x3x3',[Voigt.to_3x3x3x3_compliance,
                                       Voigt.to_3x3x3x3_stiffness,
                                       Mandel.to_3x3x3x3_compliance,
                                       Mandel.to_3x3x3x3_stiffness])
def test_squeeze_6x6_to_3x3x3x3(to_3x3x3x3):
    t = tensor.symmetric(np.random.random((6,6)))
    assert np.allclose(to_3x3x3x3(t),to_3x3x3x3(t.reshape(1,1,1,6,6)).squeeze())


@pytest.mark.parametrize('to_6x6',[Voigt.to_6x6_compliance,
                                   Voigt.to_6x6_stiffness,
                                   Mandel.to_6x6_compliance,
                                   Mandel.to_6x6_stiffness])
def test_squeeze_3x3x3x3_to_6x6(to_6x6):
    t = _conventions.convert_6x6_to_3x3x3x3(tensor.symmetric(np.random.random((6,6))),[1,1])
    assert np.allclose(to_6x6(t),to_6x6(t.reshape(1,1,1,3,3,3,3)).squeeze())



@pytest.mark.parametrize('forward,backward,rotate',
                         [(Voigt.to_6_strain,Voigt.to_3x3_strain,Voigt.rotate_strain),
                          (Voigt.to_6_stress,Voigt.to_3x3_stress,Voigt.rotate_stress),
                          (Mandel.to_6_strain,Mandel.to_3x3_strain,Mandel.rotate_strain),
                          (Mandel.to_6_stress,Mandel.to_3x3_stress,Mandel.rotate_stress)])
@pytest.mark.parametrize('t',t3x3())
def test_rotation_2nd(forward,backward,rotate,t):
    O = damask.Rotation.from_random(shape=t.shape[:-2])
    t_6 = forward(t)
    t_6_rotated = rotate(O,t_6)
    assert np.allclose(O@t,backward(t_6_rotated))

@pytest.mark.parametrize('forward,backward,rotate',
                         [(Voigt.to_6x6_compliance,Voigt.to_3x3x3x3_compliance,Voigt.rotate_compliance),
                          (Voigt.to_6x6_stiffness,Voigt.to_3x3x3x3_stiffness,Voigt.rotate_stiffness),
                          (Mandel.to_6x6_compliance,Mandel.to_3x3x3x3_compliance,Mandel.rotate_compliance),
                          (Mandel.to_6x6_stiffness,Mandel.to_3x3x3x3_stiffness,Mandel.rotate_stiffness)])
@pytest.mark.parametrize('t_6x6',t6x6())
def test_rotation_4th(forward,backward,rotate,t_6x6):
    t = _conventions.convert_6x6_to_3x3x3x3(t_6x6,[1,1])
    O = damask.Rotation.from_random(shape=t.shape[:-4])
    t_6x6 = forward(t)
    t_6x6_rotated = rotate(O,t_6x6)
    assert np.allclose(O@t,backward(t_6x6_rotated))

@pytest.mark.parametrize('stiffness_6x6_to_3x3x3x3,strain_6_to_3x3,stress_3x3_to_6',
                         [(Voigt.to_3x3x3x3_stiffness,Voigt.to_3x3_strain,Voigt.to_6_stress),
                          (Mandel.to_3x3x3x3_stiffness,Mandel.to_3x3_strain,Mandel.to_6_stress)])
@pytest.mark.parametrize('epsilon_tilde,C_tilde',t6_t6x6())
def test_stress_calculation(stiffness_6x6_to_3x3x3x3,strain_6_to_3x3,stress_3x3_to_6,epsilon_tilde,C_tilde):
    C = stiffness_6x6_to_3x3x3x3(C_tilde)
    epsilon = strain_6_to_3x3(epsilon_tilde)

    sigma = np.einsum('...ijkl,...kl',C,epsilon)
    sigma_tilde = np.einsum('...ij,...j',C_tilde,epsilon_tilde)
    assert np.allclose(stress_3x3_to_6(sigma),sigma_tilde)

@pytest.mark.parametrize('compliance_6x6_to_3x3x3x3,stress_6_to_3x3,strain_3x3_to_6',
                         [(Voigt.to_3x3x3x3_compliance,Voigt.to_3x3_stress,Voigt.to_6_strain),
                          (Mandel.to_3x3x3x3_compliance,Mandel.to_3x3_stress,Mandel.to_6_strain)])
@pytest.mark.parametrize('sigma_tilde,S_tilde',t6_t6x6())
def test_strain_calculation(compliance_6x6_to_3x3x3x3,stress_6_to_3x3,strain_3x3_to_6,sigma_tilde,S_tilde):
    S = compliance_6x6_to_3x3x3x3(S_tilde)
    sigma = stress_6_to_3x3(sigma_tilde)

    epsilon = np.einsum('...ijkl,...kl',S,sigma)
    epsilon_tilde = np.einsum('...ij,...j',S_tilde,sigma_tilde)
    assert np.allclose(strain_3x3_to_6(epsilon),epsilon_tilde)


@pytest.mark.parametrize('t_tilde',t6())
@pytest.mark.parametrize('O',O())
def test_broadcast_2nd(t_tilde,O):
    t_tilde_ = _conventions.rotate(O,t_tilde,[1.]*2,2)
    t_ = O@_conventions.convert_6_to_3x3(t_tilde,1.)
    assert t_tilde_.shape[:-1] == t_.shape[:-2]

@pytest.mark.parametrize('T_tilde',t6x6())
@pytest.mark.parametrize('O',O())
def test_broadcast_4th(T_tilde,O):
    T_tilde_ = _conventions.rotate(O,T_tilde,[1.]*2,4)
    T_ = O@_conventions.convert_6x6_to_3x3x3x3(T_tilde,[1.]*2)
    assert T_tilde_.shape[:-2] == T_.shape[:-4]
