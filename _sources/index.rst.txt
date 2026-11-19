====================
scikit-elasticity
====================

A library for working with anisotropic elasticity in Python. 


Homogenization
==============

Calculation of elastic constants from either elastic stiffness or compliance and a list of orientations.
Inputs and outputs are plain NumPy arrays in Mandel notation.
The orientation input must be a one-dimensional ``damask.Rotation``.
Hill-type estimates are available through ``C_Hill`` and ``S_Hill``; select the scheme with ``method=`` (``super_hill`` by default, also ``arithmetic_stiffness``, ``arithmetic_compliance``, ``geometric_log``, or ``geometric_affine``).
The orientation-wise geometric mean is available separately as ``C_geo`` and ``S_geo``.

.. code-block:: python

  >>> import skel
  >>> import damask
  >>> import numpy as np

  >>> N = 1000
  >>> C_Al = skel.Hooke.C('cubic',C_11=106.9e9,C_12=60.5e9,C_44=28.4e9)
  >>> C_Al = skel.Mandel.to_6x6_stiffness(skel.Voigt.to_3x3x3x3_stiffness(C_Al))
  >>> O = damask.Rotation.from_random(N)
  >>> with np.printoptions(precision=2,suppress=True):
  >>>    print(skel.homogenization.C_Reuss(O,C=C_Al)/1e9)
  >>>    print(skel.homogenization.C_Voigt(O,C=C_Al)/1e9)
  >>>    print(skel.homogenization.C_Hill(O,C=C_Al,method='arithmetic_stiffness')/1e9)
  >>>    print(skel.homogenization.C_Hill(O,C=C_Al,method='arithmetic_compliance')/1e9)

.. automodule:: skel.homogenization
   :members:

Hooke
=====

.. automodule:: skel.Hooke
   :members:


Conversions and Representations
===============================

Voigt
-----

.. automodule:: skel.Voigt
   :members:

|

Mandel
------

.. automodule:: skel.Mandel
   :members:

|

natural
-------

.. automodule:: skel.natural
   :members:



.. toctree::
   :maxdepth: 2
   :caption: Contents:
