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
         'C_11': 283.0e+9, 'C_12':154.4e+9, 'C_13':126.2e+9, 'C_22':273.6e+9, 'C_23':126.8e+9,
         'C_33': 279.8e+9, 'C_44':72.4e+9, 'C_55':47.3e+9, 'C_66':92.0e+9}                          # https://doi.org/10.1088/2053-1591/aab118

examples = [iso,Fe,Mg,corundum,Sn_beta,Al2O2]


optional_zero = {
    'trigonal':[(0,4),(1,4),(3,5)],
    'tetragonal':[(0,5),(1,5)]
}

condition_equal = {
    'isotropic': {'11':['22','33'],
                  '12':['13','23']},
    'cubic': {'11':['22','33'],
              '12':['13','23'],
              '44':['55','66']},
    'hexagonal': {'11':['22'],
                  '13':['23'],
                  '44':['55']},
    'trigonal': {'11':['22'],
                 '13':['23'],
                 '44':['55']},
    'tetragonal': {'11':['22'],
                   '13':['23'],
                   '44':['55']},
}


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


@pytest.mark.parametrize('material',examples)
def test_eigenvalues(material):
    C = Hooke.C(**material)
    assert np.all(np.linalg.eigvalsh(C)>0)

@pytest.mark.parametrize('material',examples)
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

@pytest.mark.parametrize('material',examples)
def test_initialize_S(material):
    S = np.linalg.inv(Hooke.C(**material))
    material_mut = material.copy()
    m = {'crystal_system':material_mut.pop('crystal_system')}
    for k in material_mut.keys():
        idx = k.split('_')[1]
        m[f'S_{idx}'] = S[int(idx[0])-1,int(idx[1])-1]
    f = 1./np.max(np.abs(S))
    assert np.allclose(S*f,Hooke.S(**m)*f)

@pytest.mark.parametrize('material',[iso,Fe,Mg,corundum,Sn_beta])
def test_initialize_alternatives(material,np_rng):
    material_mut = material.copy()
    for defined,alt in condition_equal[material_mut['crystal_system']].items():
        val = material_mut[f'C_{defined}']
        if (N_add := np_rng.integers(len(alt)+1)) == 0: continue
        for a in (add := np_rng.choice(alt,size=N_add,replace=False)):
            material_mut[f'C_{a}'] = val
        if (N_rm := np_rng.integers(N_add+1)) == 0: continue
        for r in np_rng.choice(add.tolist()+[defined],size=N_rm,replace=False):
            del material_mut[f'C_{r}']
    assert np.allclose(Hooke.C(**material_mut),Hooke.C(**material))

@pytest.mark.parametrize('material',[iso,Fe,Mg,corundum,Sn_beta])
def test_initialize_invalid(material,np_rng):
    material_mut = material.copy()
    for defined,alt in condition_equal[material_mut['crystal_system']].items():
        val = material_mut[f'C_{defined}']
        N_add = np_rng.integers(1,len(alt)+1)
        for a in (add := np_rng.choice(alt,size=N_add,replace=False)):
            material_mut[f'C_{a}'] = val * np_rng.choice([0.99,1.01])
    with pytest.raises(ValueError):
        Hooke.C(**material_mut)

def test_isotropic_Youngs_modulus(np_rng):
    N = np_rng.integers(1,11)
    S = np.linalg.inv(Hooke.C(**iso))
    x = damask.Rotation.from_random(N,rng_seed=np_rng)@np.array([1,0,0])
    assert np.allclose(1./S[0,0],Hooke.E(x,S=S))

def test_isotropic_shear_modulus(np_rng):
    N = np_rng.integers(1,11)
    S = np.linalg.inv(Hooke.C(**iso))
    h_u = damask.Rotation.from_random(N,rng_seed=np_rng).as_matrix()[:,:2]
    assert np.allclose(1./S[3,3],Hooke.G(h_u[:,0],h_u[:,1],S=S))

def test_isotropic_Poisson_ratio(np_rng):
    N = np_rng.integers(1,11)
    S = np.linalg.inv(Hooke.C(**iso))
    x_y = damask.Rotation.from_random(N,rng_seed=np_rng).as_matrix()[:,:2]
    assert np.allclose(0.5*S[3,3]/S[0,0]-1,Hooke.nu(x_y[:,0],x_y[:,1],S=S))


def test_anisotropy_iso(np_rng):
    C_11 = np_rng.uniform(50e9,300e9)
    C_12 = np_rng.uniform(-C_11*.45,C_11*.95)
    C = Hooke.C('isotropic',C_11=C_11,C_12=C_12)
    assert np.allclose(Hooke.A_u(C=C),0.0)

def test_anisotropy_cubic(np_rng):
    C_11 = np_rng.uniform(50e9,300e9)
    C_12 = np_rng.uniform(-C_11*.45,C_11*.95)
    C_44 = np_rng.uniform(0,300e9)
    C = Hooke.C('cubic',C_11=C_11,C_12=C_12,C_44=C_44)
    A = 2*C[3,3]/(C[0,0]-C[0,1])
    assert np.allclose(Hooke.A_u(C=C),6./5.*(A**.5-A**(-.5))**2)

@pytest.mark.parametrize('material',examples)
def test_anisotropy_rotation(material,np_rng):
    C = np.linalg.inv(Hooke.C(**material))
    assert np.allclose(Hooke.A_u(C=C),Hooke.A_u(C=Voigt.rotate_stiffness(damask.Rotation.from_random(),C)))
