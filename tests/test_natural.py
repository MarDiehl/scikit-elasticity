# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import damask
from damask import tensor

from skel import Hooke
from skel import Mandel
from skel import Voigt
from skel import natural


def deviatoric(t):
    """Deviatoric part, compatible with DAMASK versions before/after `traceless`."""
    return tensor.traceless(t) if hasattr(tensor,'traceless') else tensor.deviatoric(t)


def test_basis_orthonormal(assert_allclose):
    E = natural._basis(0.0)
    assert_allclose(np.einsum('aij,bij',E,E),np.eye(6))


def test_basis_orthonormal_rotated(np_rng,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    E = natural._basis(theta)
    assert_allclose(np.einsum('aij,bij',E,E),np.eye(6))


def test_theta_roundtrip_2nd(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,5))
    assert_allclose(natural.to_3x3(natural.to_6(t,theta),theta),t)


def test_theta_roundtrip_4th(np_rng,symmetric_4th,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    T = symmetric_4th((3,))
    assert_allclose(natural.to_3x3x3x3(natural.to_6x6(T,theta),theta),T)


def test_theta_does_not_change_invariants(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))
    t_6 = natural.to_6(t)
    t_6_rotated = natural.to_6(t,theta)
    assert_allclose(np.linalg.norm(t_6_rotated,axis=-1),np.linalg.norm(t_6,axis=-1))
    assert_allclose(t_6_rotated[...,0],t_6[...,0])


def test_hydrostatic_component(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))
    assert_allclose(natural.to_6(t,theta)[...,0]/np.sqrt(3),tensor.spherical(t)[...,0,0])


def test_spherical_component(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))

    p = tensor.spherical(t,tensor=False)
    t_6 = natural.to_6(tensor.spherical(t),theta)
    assert_allclose(t_6[...,0],np.sqrt(3.0)*p)
    assert_allclose(t_6[...,1:],0.0)


def test_deviatoric_norm(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))
    t_dev = deviatoric(t)
    t_6 = natural.to_6(t_dev,theta)
    assert_allclose(t_6[...,0],0.0)
    assert_allclose(np.linalg.norm(t_6[...,1:],axis=-1),
                       np.linalg.norm(t_dev,axis=(-2,-1)))


def test_hydrostatic_deviatoric_decomposition(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))

    t_h = tensor.spherical(t)
    t_d = deviatoric(t)

    assert_allclose(t_h+t_d,t)
    assert_allclose(np.einsum('...ij,...ij',t_h,t_d),0.0)
    assert_allclose(natural.to_6(t,theta),
                       natural.to_6(t_h,theta)+natural.to_6(t_d,theta))


def test_shear_components_match_mandel(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))
    assert_allclose(natural.to_6(t,theta)[...,3:],Mandel.to_6(t)[...,3:])


def test_stiffness_and_compliance_inverse(np_rng,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    C66 = Hooke.C('cubic',C_11=232.2e9,C_12=136.4e9,C_44=117.0e9)
    C = Voigt.to_3x3x3x3_stiffness(C66)
    S = Voigt.to_3x3x3x3_compliance(np.linalg.inv(C66))
    assert_allclose(natural.to_6x6_stiffness(C,theta)@natural.to_6x6_compliance(S,theta),
                       np.eye(6))


def test_eigenvalues_match_mandel(np_rng,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    C = Voigt.to_3x3x3x3_stiffness(Hooke.C('cubic',C_11=232.2e9,C_12=136.4e9,C_44=117.0e9))
    assert_allclose(np.linalg.eigvalsh(natural.to_6x6_stiffness(C,theta)),
                       np.linalg.eigvalsh(Mandel.to_6x6_stiffness(C)))


def test_isotropic_stiffness(np_rng,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    K,G = 120e9,80e9
    C = Voigt.to_3x3x3x3_stiffness(Hooke.C('isotropic',
                                          C_11=K+4*G/3,
                                          C_12=K-2*G/3))
    assert_allclose(natural.to_6x6_stiffness(C,theta)/1e9,np.diag([3*K]+[2*G]*5)/1e9)


def test_isotropic_response(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    K,G = 120e9,80e9
    C = Voigt.to_3x3x3x3_stiffness(Hooke.C('isotropic',
                                          C_11=K+4*G/3,
                                          C_12=K-2*G/3))

    epsilon = symmetric_2nd((4,))
    sigma = np.einsum('...ijkl,...kl',C,epsilon)

    epsilon_6 = natural.to_6_strain(epsilon,theta)
    sigma_6 = natural.to_6_stress(sigma,theta)

    assert_allclose(sigma_6[...,0],3*K*epsilon_6[...,0])
    assert_allclose(sigma_6[...,1:],2*G*epsilon_6[...,1:])


def test_stress_calculation(np_rng,symmetric_4th,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    C = symmetric_4th((4,))
    epsilon = symmetric_2nd((4,))

    sigma = np.einsum('...ijkl,...kl',C,epsilon)
    sigma_natural = np.einsum('...ij,...j',
                              natural.to_6x6_stiffness(C,theta),
                              natural.to_6_strain(epsilon,theta))
    assert_allclose(natural.to_6_stress(sigma,theta),sigma_natural)


def test_strain_calculation(np_rng,symmetric_4th,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    S = symmetric_4th((4,))
    sigma = symmetric_2nd((4,))

    epsilon = np.einsum('...ijkl,...kl',S,sigma)
    epsilon_natural = np.einsum('...ij,...j',
                                natural.to_6x6_compliance(S,theta),
                                natural.to_6_stress(sigma,theta))
    assert_allclose(natural.to_6_strain(epsilon,theta),epsilon_natural)


def test_rotation_2nd(np_rng,symmetric_2nd,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((4,))
    O = damask.Rotation.from_random(t.shape[:-2],np_rng)
    assert_allclose(natural.rotate_strain(O,natural.to_6(t,theta),theta),
                       natural.to_6(O@t,theta))


def test_rotation_4th(np_rng,symmetric_4th,assert_allclose):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    T = symmetric_4th((4,))
    O = damask.Rotation.from_random(T.shape[:-4],np_rng)
    assert_allclose(natural.rotate_stiffness(O,natural.to_6x6(T,theta),theta),
                       natural.to_6x6(O@T,theta))


def test_rotation_broadcast_2nd(np_rng,symmetric_2nd):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    t = symmetric_2nd((3,4))
    O = damask.Rotation.from_random(3,np_rng)
    assert natural.rotate_stress(O,natural.to_6(t,theta),theta).shape == (3,4,6)


def test_rotation_broadcast_4th(np_rng,symmetric_4th):
    theta = np_rng.uniform(-4*np.pi,4*np.pi)
    T = symmetric_4th((3,4))
    O = damask.Rotation.from_random(3,np_rng)
    assert natural.rotate_compliance(O,natural.to_6x6(T,theta),theta).shape == (3,4,6,6)
