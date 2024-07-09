# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
from scipy import linalg, optimize
from . import Mandel
from . import Voigt
from . import Eshelby

class VolumeElement():
    def __init__(self,C,O):
        self.C = np.reshape(Voigt.rotate_stiffness(O,C),(-1,6,6))
        self.S = np.reshape(Voigt.rotate_compliance(O,np.linalg.inv(C)),(-1,6,6))

# Voigt
    @property
    def C_Voigt(self):
        """Homogenized stiffness according to Voigt."""
        return np.average(self.C,axis=0)
    @property
    def S_Voigt(self):
        """Homogenized compliance according to Voigt."""
        return np.linalg.inv(self.C_Voigt)


# Reuss
    @property
    def S_Reuss(self):
        """Homogenized stiffness according to Reuss."""
        return np.average(self.S,axis=0)
    @property
    def C_Reuss(self):
        """Homogenized compliance according to Reuss."""
        return np.linalg.inv(self.S_Reuss)


# Hill
    @property
    def C_Hill_C(self):
        """
        Homogenized stiffness according to Hill.

        Average of C from Voigt and Reuss.
        """
        return (self.C_Voigt+self.C_Reuss)*.5
    @property
    def S_Hill_S(self):
        """
        Homogenized compliance according to Hill.

        Average of S from Voigt and Reuss.
        """
        return (self.S_Voigt+self.S_Reuss)*.5

    @property
    def C_Hill_S(self):
        """
        Homogenized stiffness according to Hill.

        Average of S from Voigt and Reuss.
        """
        return np.linalg.inv(self.S_Hill_S)
    @property
    def S_Hill_C(self):
        """
        Homogenized compliance according to Hill.

        Average of C from Voigt and Reuss.
        """
        return np.linalg.inv(self.C_Hill_C)

    @property
    def C_Hill(self):
        """
        Homogenized stiffness according to Hill.

        Average of C and S average from Voigt and Reuss.
        """
        return (self.C_Hill_C+self.C_Hill_S)*.5
    @property
    def S_Hill(self):
        """
        Homogenized compliance according to Hill.

        Average of S and C average from Voigt and Reuss.
        """
        return (self.S_Hill_S+self.S_Hill_C)*.5

    @property
    def C_Hill_geometric(self):
        """Homogenized stiffness according to Hill."""
        return linalg.sqrtm(self.C_Voigt@self.C_Reuss)
    @property
    def S_Hill_geometric(self):
        """Homogenized compliance according to Hill."""
        return linalg.sqrtm(self.S_Voigt@self.S_Reuss)


    @property
    def C_sc(self):
        # https://dx.doi.org/10.1002/zamm.201100135, eq. (4)
        def f(x,xis,w,C_II):
            E = Mandel.to_6x6(Eshelby.eshelby(Mandel.to_3x3x3x3(x),xi=xis,w=w))@np.linalg.inv(x)
            return np.average(C_II@np.linalg.inv(E@(C_II-x)+np.eye(6)),axis=0)

        xis, w = Eshelby._xi([1,1,1],8,64)
        C_II = Mandel.to_6x6(Voigt.to_3x3x3x3_stiffness(self.C))
        C_I = Mandel.to_6x6(Voigt.to_3x3x3x3_stiffness(self.C_Hill))

        C = optimize.fixed_point(f,C_I,args=(xis,w,C_II),xtol=1e-4)
        return Voigt.to_6x6_stiffness(Mandel.to_3x3x3x3_stiffness(C))
