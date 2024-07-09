# scikit-elasticity

A library for working with anisotropic elasticity in Python

## Features
All functionality is implemented in "vectorized" form, i.e. multidimensional arrays are handled without explicit loops in Python.
Hence, despited being a pure Python library (if the dependency on NumPy and SciPy is ignored), it is suitable for large scale calculations.

### Homogenization
Calculation of effective stiffness

### Conversions
Represent 2nd and 4th order tensors in compressed Voigt or Mandel notation.
Including support for rotations, which is based on the Python implementation of the conventions by Rowenhorst et al.

### Stiffness or Compliance Matrices
Initialize stiffness or compliance matrices for different crystal systems.


## Installation

### Development
From the base directory run `export PYTHONPATH=:$PWD/skel:$PYTHONPATH`.

To get the correct version, also run `versioningit -w` from the root directory of the package.
This requires to have [versioningit](https://github.com/jwodder/versioningit) installed.

### From source
```
python -m pip install .
```

## Usage
```python
import skel
import damask
import numpy as np

N = 100000
C_Fe = skel.Hooke.C('cubic',C_11=232.2e9,C_12=136.4e9,C_44=117.0e9)
C_Al = skel.Hooke.C('cubic',C_11=106.9e9,C_12=60.5e9,C_44=28.4e9)
C_Cu = skel.Hooke.C('cubic',C_11=168.9e9,C_12=121.8e9,C_44=75.8e9)
C_Mg = skel.Hooke.C('hexagonal',C_11=59.5e9,C_12=25.6e9,C_44=16.5e9,C_33=61.7e9,C_13=21.5e9)
O = damask.Rotation.from_random(N,rng_seed=20191102)
for label,C in zip(['Fe','Al','Cu','Mg'],[C_Fe,C_Al,C_Cu,C_Mg]):
    print(f'== {label} ==')
    ve = skel.VolumeElement(C,O)
    with np.printoptions(precision=2,suppress=True):
        print('Reuss\n',ve.C_Reuss/1e9)
        print('Voigt\n',ve.C_Voigt/1e9)
        print('Hill\n',ve.C_Hill/1e9)
        print('Self-consistent\n',ve.C_sc/1e9)

# https://www.sciencedirect.com/science/article/abs/pii/0956715195901698 (fig 6)
# https://doi.org/10.1080/02670836.2016.1231746
def texture_Ni(N_grains,rng):
    Eulers = [[[59,37,63],.30], # S
              [[90,35,45],.20], # Cu
              [[35,45,90],.15]] # Brass
    directions = [[[np.pi/4.,0.],[0.,0.],.10]]
    O = damask.Rotation.from_random(N_grains \
                                    - sum(int(N_grains*_[1]) for _ in Eulers)\
                                    - sum(int(N_grains*_[2]) for _ in directions),rng_seed=rng)
    for eu,v in Eulers:
        p = damask.Orientation.from_Euler_angles(phi=eu,degrees=True,lattice='cI')
        O = O.append(damask.Rotation.from_spherical_component(p,50,int(N_grains*v),degrees=True,rng_seed=rng))
    for c,s,v in directions:
        O = O.append(damask.Rotation.from_fiber_component(c,s,np.deg2rad(20),shape=int(N_grains*v),rng_seed=rng))
    return O

C_Ni = skel.Hooke.C('cubic',C_11=251.0e+9,C_12=150.0e+9,C_44=123.7e+9)
ve = skel.VolumeElement(C_Ni,texture_Ni(2000,2))
C = skel.Voigt.to_3x3x3x3_compliance(np.linalg.inv(ve.C_sc))
print('E_x',skel.Hooke.E(C,[1,0,0])/1e9)
print('E_y',skel.Hooke.E(C,[0,1,0])/1e9)
print('E_z',skel.Hooke.E(C,[0,0,1])/1e9)
```

## Contact
Martin Diehl  
KU Leuven  
martin.diehl@kuleuven.be
