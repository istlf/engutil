import numpy as np
import engutil

"""
USAGE:

Cmatrix = np.array([
    [8.501e-11, -2.173e-12],
    [-2.173e-12, 8.501e-11]
])

Lmatrix = np.array([
    [3.592e-7, 3.218e-8],
    [3.218e-8, 3.592e-7]
])

slem = engutil.SLEM(Cmatrix=Cmatrix, Lmatrix=Lmatrix)

slem.Z0_even

"""

class SLEM:
    def __init__(self, Cmatrix, Lmatrix):
        self.C = np.array(Cmatrix)
        self.L = np.array(Lmatrix)
    
    # Based on section 4.2.2 in high speed book
    @property
    def CM(self):
        return np.abs(self.C[0][1])
    @property
    def Cg(self):
        return self.C[0][0] - self.CM
    @property
    def L0(self):
        return self.L[0][0]
    @property
    def LM(self):
        return self.L[0][1]
    
    @property
    def Z0_isolated(self):
        r"""
        equation 4.35

        Z_{0,isolated} = \sqrt{\frac{L_{0}}{C_{g} + C_{M}}}

        """
        return np.sqrt(self.L0/(self.Cg + self.CM))
    
    @property
    def Z0_even(self):
        r"""
        equation 4.36
        Z_{0,even} = \sqrt{\frac{L_{0} + L_{M}}{C_{g}}}
        """
        return np.sqrt((self.L0 + self.LM) / self.Cg)

    @property
    def Z0_odd(self):
        r"""
        equation 4.37
        Z_{0,odd} = \sqrt{\frac{L_{0} - L_{M}}{C_{g} + 2C_{M}}}
        """
        return np.sqrt((self.L0 - self.LM) / (self.Cg + 2 * self.CM))

    @property
    def vp_isolated(self):
        r"""
        equation 4.38
        v_{p,isolated} = \frac{1}{\sqrt{L_{0}(C_{g} + C_{M})}}
        """
        return 1 / np.sqrt(self.L0 * (self.Cg + self.CM))

    @property
    def vp_even(self):
        r"""
        equation 4.39
        v_{p,even} = \frac{1}{\sqrt{(L_{0} + L_{M})C_{g}}}
        Note: Image 4-39 shows C_0, likely a typo for C_g for consistency with 4-36.
        """
        return 1 / np.sqrt((self.L0 + self.LM) * self.Cg)

    @property
    def vp_odd(self):
        r"""
        equation 4.40
        v_{p,odd} = \frac{1}{\sqrt{(L_{0} - L_{M})(C_{g} + 2C_{M})}}
        """
        return 1 / np.sqrt((self.L0 - self.LM) * (self.Cg + 2 * self.CM))