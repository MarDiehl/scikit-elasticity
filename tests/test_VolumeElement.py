# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import damask

import skel

def test_equivalent(assert_allclose):
    C_11 = 124.1e9
    C_12 = 93.7e9
    C_44 = 46.4e9
    C = skel.Hooke.C('cubic',C_11=C_11,C_12=C_12,C_44=C_44)
    ve = skel.VolumeElement(C,damask.Rotation())
    assert_allclose(C,ve.C_Reuss)
    assert_allclose(C,ve.C_Voigt)
    assert_allclose(C,ve.C_Hill)
    assert_allclose(C,ve.C_Hill_geometric)
    assert_allclose(C,ve.C_Hill_C)
    assert_allclose(C,ve.C_Hill_S)

    assert_allclose(C@ve.S_Reuss,np.eye(6))
    assert_allclose(C@ve.S_Voigt,np.eye(6))
    assert_allclose(C@ve.S_Hill,np.eye(6))
    assert_allclose(C@ve.S_Hill_geometric,np.eye(6))
    assert_allclose(C@ve.S_Hill_S,np.eye(6))
    assert_allclose(C@ve.S_Hill_C,np.eye(6))
