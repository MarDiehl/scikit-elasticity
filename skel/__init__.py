# SPDX-License-Identifier: AGPL-3.0-or-later
# https://ilab.imr.tohoku.ac.jp/ourtexts/micromechanics/MicromechanicsApplicationhtmlLyXconv/MicromechanicsApplication.html
# https://github.com/ddempsey/eshelby-inclusion/blob/master/elpinc.py
# https://github.com/Chunfang/Esh3D
# https://eprints.maths.manchester.ac.uk/2353/1/hill_tensor_review.pdf
# https://gitlab.onelab.info/cm3/cm3MFH/-/blob/4686d5a591fd2f21385c38c235855606111e06e5/op_eshelby.f
# https://progs.coudert.name/elate
# https://engineering.stackexchange.com/questions/16185
# https://github.com/romerogroup/MechElastic
# https://progs.coudert.name/elate
# https://elasticipy.readthedocs.io
# https://rockphypy.readthedocs.io
# https://scema.mpie.de/
# https://jfbarthelemy.github.io/echoes
from pathlib import Path as _Path
from importlib.metadata import version as _version, PackageNotFoundError
import re as _re

name = 'scikit-elasticity'
try:
    with open(_Path(__file__).parent.parent/_Path('VERSION')) as _f:
        __version__ = _re.sub(r'^v','',_f.readline().strip())
except FileNotFoundError:
    try:
        # ToDo: Check if found package is equivalent to used one
        __version__ = _version('scikit-elasticity')
    except PackageNotFoundError:
        __version__ = 'unknown'

from . import Voigt
from . import Mandel
from . import Hooke
from . import Eshelby
from ._volumeelement import VolumeElement
