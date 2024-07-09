# SPDX-License-Identifier: AGPL-3.0-or-later
import damask
import numpy as np
import pytest

from skel import Hooke, Voigt

iso = {'crystal_system':'isotropic',
       'C_11':278.2e+9, 'C_12':113.4e+9}

Fe = {'crystal_system':'cubic',
      'C_11':232.2e+9, 'C_12':136.4e+9, 'C_44':117.0e+9}                                            # from DAMASK examples

Mg = {'crystal_system':'hexagonal',
      'C_11':59.5e+9, 'C_12':25.6e+9, 'C_44':16.5e+9, 'C_33':61.7e+9, 'C_13':21.5e+9}               # from DAMASK examples

corundum = {'crystal_system':'trigonal',
            'C_11':496.9e+9, 'C_33':500.5e+9, 'C_44':146.8e+9, 'C_12':162.e+9, 'C_13':115.5e+9,
            'C_14':-21.9e+9}                                                                        # https://doi.org/10.1016/0022-3697(86)90141-1

Sn_beta = {'crystal_system':'tetragonal',
           'C_11':72.6e+9, 'C_12':59.4e+9, 'C_44':22.2e+9, 'C_33':88.8e+9, 'C_13':35.8e+9,
           'C_66':24.1e+9}                                                                          # from DAMASK examples

Al2O2 = {'crystal_system':'orthorhombic',
         'C_11': 283.e+9, 'C_12':154.4e+9, 'C_13':126.2e+9, 'C_22':273.6e+9, 'C_23':126.8e+9,
         'C_33': 279.8e+9, 'C_44':72.4e+9, 'C_55':47.3e+9, 'C_66':92.0e+9}                           # https://doi.org/10.1088/2053-1591/aab118

optional_zero = {'trigonal':[(0,4),(1,4),(3,5)],
                 'tetragonal':[(0,5),(1,5)]}

def invert_cubic(X_11,X_12,X_44):
    # https://srd.nist.gov/jpcrdreprint/1.3253127.pdf, Tab 3
    return ((X_11+X_12)/((X_11-X_12)*(X_11+2*X_12)),
            -X_12/((X_11-X_12)*(X_11+2*X_12)),
            1/X_44)


def test_inverse_cubic():
    C = Hooke.C(**Fe)
    S_11,S_12,S_44 = invert_cubic(Fe['C_11'],Fe['C_12'],Fe['C_44'])
    S = Hooke.S('cubic',S_11=S_11,S_12=S_12,S_44=S_44)
    f = 1./np.max(np.abs(S))
    assert np.allclose(np.linalg.inv(C)*f,S*f)


@pytest.mark.parametrize('material',[Fe,Mg,corundum,Sn_beta,Al2O2])
def test_eigenvalues(material):
    C = Hooke.C(**material)
    assert np.all(np.linalg.eigvalsh(C)>0)

@pytest.mark.parametrize('material',[Fe,Mg,corundum,Sn_beta,Al2O2])
def test_enforce_zero(material,np_rng):
    C = Hooke.C(**material) + np.tril(np.ones(6),-1)
    if (mask := optional_zero.get(material['crystal_system'],None)) is not None:
        C[mask] = 1
    r,c = np.nonzero(C==0)
    i = np_rng.integers(len(r))
    material_mut = material.copy()
    material_mut[f'C_{r[i]+1}{c[i]+1}'] = material['C_11']
    with pytest.raises(ValueError):
        Hooke.C(**material_mut)

def test_isotropic_Youngs_modulus(np_rng):
    N = np_rng.integers(1,11)
    S = np.linalg.inv(Hooke.C(**iso))
    x = damask.Rotation.from_random(N,rng_seed=np_rng)@np.array([1,0,0])
    assert np.allclose(1./S[0,0],Hooke.E(Voigt.to_3x3x3x3_compliance(S),x))

def test_isotropic_shear_modulus(np_rng):
    N = np_rng.integers(1,11)
    S = np.linalg.inv(Hooke.C(**iso))
    h_u = damask.Rotation.from_random(N,rng_seed=np_rng).as_matrix()[:,:2]
    assert np.allclose(1./S[3,3],Hooke.G(Voigt.to_3x3x3x3_compliance(S),h_u[:,0],h_u[:,1]))

def test_isotropic_Poisson_ratio(np_rng):
    N = np_rng.integers(1,11)
    S = np.linalg.inv(Hooke.C(**iso))
    x_y = damask.Rotation.from_random(N,rng_seed=np_rng).as_matrix()[:,:2]
    assert np.allclose(0.5*S[3,3]/S[0,0]-1,Hooke.nu(Voigt.to_3x3x3x3_compliance(S),x_y[:,0],x_y[:,1]))


def test_anisotropy_iso(np_rng):
    C_11 = np_rng.uniform(50e9,300e9)
    C_12 = np_rng.uniform(-C_11*.45,C_11*.95)
    C = Hooke.C('isotropic',C_11=C_11,C_12=C_12)
    S = np.linalg.inv(C)
    assert np.allclose(Hooke.A_u(C=C,S=S),0.0)

def test_anisotropy_cubic(np_rng):
    C_11 = np_rng.uniform(50e9,300e9)
    C_12 = np_rng.uniform(-C_11*.45,C_11*.95)
    C_44 = np_rng.uniform(0,300e9)
    C = Hooke.C('cubic',C_11=C_11,C_12=C_12,C_44=C_44)
    S = np.linalg.inv(C)
    A = 2*C[3,3]/(C[0,0]-C[0,1])
    assert np.allclose(Hooke.A_u(C=C,S=S),6./5.*(A**.5-A**(-.5))**2)
