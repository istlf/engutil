import numpy as np 
def to_cartesian(polar_tuple):
    """
    Converts a (magnitude, angle_in_degrees) tuple into a complex number (a + jb).
    """
    mag, angle_deg = polar_tuple
    return mag * np.exp(1j * np.deg2rad(angle_deg))

def to_polar(complex_val):
    """
    Converts a complex number (a + jb) into a (magnitude, angle_in_degrees) tuple.
    """
    mag = np.abs(complex_val)
    angle_deg = np.rad2deg(np.angle(complex_val))
    return (mag, angle_deg)

class TwoPortNetwork:
    def __init__(self, s_matrix, gamma_opt):
        self.s = np.array(s_matrix, dtype=complex)
        self.gamma_op = gamma_opt
        
    @property
    def S11(self): return self.s[0, 0]
    @property
    def S12(self): return self.s[0, 1]
    @property
    def S21(self): return self.s[1, 0]
    @property
    def S22(self): return self.s[1, 1]
    
    @property
    def delta(self):
        """Determinant of the S-matrix"""
        return (self.S11 * self.S22) - (self.S12 * self.S21)
        
    @property
    def K(self):
        """Rollett Stability Factor (K > 1 and |delta| < 1 indicates unconditional stability)"""
        num = 1 - np.abs(self.S11)**2 - np.abs(self.S22)**2 + np.abs(self.delta)**2
        den = 2 * np.abs(self.S12 * self.S21)
        return num / den

    @property
    def MSG(self):
        return 10*np.log10(np.abs(self.S21)/np.abs(self.S12))
        

    @property
    def MAG(self):
        return np.abs(self.S21)/np.abs(self.S12)*(self.K - np.sqrt(self.K**2 - 1))


    def get_source_stability_circle(self):
        """Calculates Center and Radius for the Source Stability Circle (Gamma_S plane)"""
        D = self.delta
        # Formula for C_S and R_S
        C_s = np.conj(self.S11 - D * np.conj(self.S22)) / (np.abs(self.S11)**2 - np.abs(D)**2)
        R_s = np.abs(self.S12 * self.S21) / np.abs(np.abs(self.S11)**2 - np.abs(D)**2)
        return C_s, R_s

    def get_load_stability_circle(self):
        """Calculates Center and Radius for the Load Stability Circle (Gamma_L plane)"""
        D = self.delta
        # Formula for C_L and R_L
        C_l = np.conj(self.S22 - D * np.conj(self.S11)) / (np.abs(self.S22)**2 - np.abs(D)**2)
        R_l = np.abs(self.S12 * self.S21) / np.abs(np.abs(self.S22)**2 - np.abs(D)**2)
        return C_l, R_l



    @staticmethod
    def generate_circle_locus(center, radius, num_points=200):
        """Generates complex points to draw a circle"""
        theta = np.linspace(0, 2 * np.pi, num_points)
        return center + radius * np.exp(1j * theta)

def calc_transducer_gain(S21, S22, Gamma_s, Gamma_L, Gamma_in):
    """
    Calculates Transducer Power Gain (G_T) based on the standard RF formula.
    Returns the gain as a linear ratio (not dB).
    """
    # 1st term: Source match effect
    term_source = (1 - np.abs(Gamma_s)**2) / np.abs(1 - Gamma_s * Gamma_in)**2
    
    # 2nd term: Forward gain of the transistor
    term_transistor = np.abs(S21)**2
    
    # 3rd term: Load match effect
    term_load = (1 - np.abs(Gamma_L)**2) / np.abs(1 - S22 * Gamma_L)**2
    
    G_T = term_source * term_transistor * term_load
    
    return G_T


