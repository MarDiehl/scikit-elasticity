scikit-elasticity
====================

A library for working with anisotropic elasticity in Python. 


==============
Homogenization
==============

Calculation of elastic constants from elastic stiffness and a list of orientations.

.. code-block:: python

  >>> import skel
  >>> import damask
  >>> import numpy as np

  >>> N = 1000
  >>> C_Al = skel.stiffness('cubic',C_11=106.9e9,C_12=60.5e9,C_44=28.4e9)
  >>> O = damask.Rotation.from_random(N)
  >>> p = skel.Polycrystal(C_Al,O)
  >>> with np.printoptions(precision=2,suppress=True):
  >>>    print(p.C_Reuss/1e9)
  >>>    print(p.C_Voigt/1e9)
  >>>    print(p.C_Hill_C/1e9)
  >>>    print(p.C_Hill_S/1e9)
 

.. autoclass:: skel.VolumeElement
   :members:

===============================
Conversions and Representations
===============================

.. automodule:: skel.Voigt
   :members:

.. automodule:: skel.Mandel
   :members:



.. toctree::
   :maxdepth: 2
   :caption: Contents:
