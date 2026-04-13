import numpy as np
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Circle:
    center: complex
    radius: float
    label: str = ""
    def points(self):
        theta = np.linspace(0, 2 * np.pi, 200)
        return self.center + self.radius * np.exp(1j * theta)

class TwoPortNetwork:
    def __init__(self, s_matrix, noise_params=None, z0=50):
        """
        noise_params: dict with keys {'Fmin_dB', 'Rn', 'gamma_opt'}
        """
        self.s = np.array(s_matrix, dtype=complex)
        self.z0 = z0
        self.noise_params = noise_params
    
    # --- Basic S-Parameter Properties ---
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
        return (self.S11 * self.S22) - (self.S12 * self.S21)
        
    @property
    def K(self):
        num = 1 - np.abs(self.S11)**2 - np.abs(self.S22)**2 + np.abs(self.delta)**2
        den = 2 * np.abs(self.S12 * self.S21)
        return num / den

    # --- Gain Properties ---
    @property
    def max_stable_gain_db(self):
        return 10 * np.log10(np.abs(self.S21) / np.abs(self.S12))

    # --- Stability Circles ---
    @property
    def source_stability_circle(self) -> Circle:
        D = self.delta
        den = (np.abs(self.S11)**2 - np.abs(D)**2)
        c = np.conj(self.S11 - D * np.conj(self.S22)) / den
        r = np.abs(self.S12 * self.S21) / np.abs(den)
        return Circle(c, r, "Source Stability")

    @property
    def load_stability_circle(self) -> Circle:
        D = self.delta
        den = (np.abs(self.S22)**2 - np.abs(D)**2)
        c = np.conj(self.S22 - D * np.conj(self.S11)) / den
        r = np.abs(self.S12 * self.S21) / np.abs(den)
        return Circle(c, r, "Load Stability")

    # --- Constant Gain Circles (Available Gain) ---
    def available_gain_circle(self, gain_db: float) -> Circle:
        """Calculates the Ga circle for the Source plane (Gamma_S) - page 257 in Gonzales"""
        Ga =  10**(gain_db / 10)
        ga = Ga / (np.abs(self.S21)**2)
        
        c1 = self.S11 - self.delta * np.conj(self.S22)
        den = 1 + ga * (np.abs(self.S11)**2 - np.abs(self.delta)**2)
        
        ca = (ga * np.conj(c1)) / den
        
        s12s21 = np.abs(self.S12 * self.S21)
        num_r = np.sqrt(1 - 2 * self.K * s12s21 * ga + (s12s21 * ga)**2)
        ra = num_r / np.abs(den)
        
        return Circle(ca, ra, f"Ga={gain_db}dB")


# def available_gain_circle(S11, S12, S21, S22, Ga, K, delta):
#     """
#     Calculates the center (Ca) and radius (ra) of available gain circles 
#     using the provided S-parameters and gain factor.
#     """
    
#     ga = Ga/(np.abs(S21)**2)

#     c1 = S11 - delta * np.conj(S22)
    
#     denominator = 1 + ga * (np.abs(S11)**2 - np.abs(delta)**2)
    
#     ca = (ga * np.conj(c1)) / denominator
    
#     s12_s21_mag = np.abs(S12 * S21)
    
#     ra_numerator = np.sqrt(1 - 2 * K * s12_s21_mag * ga + (s12_s21_mag * ga)**2)
    
#     ra = ra_numerator / np.abs(denominator)
    
#     return ca, ra



    # --- Noise Figure Circles ---
    def noise_circle(self, F_target_db: float) -> Circle:
        if not self.noise_params:
            raise ValueError("Noise parameters (Fmin_dB, Rn, gamma_opt) not provided.")
        
        F_min = 10**(self.noise_params['Fmin_dB'] / 10)
        F_target = 10**(F_target_db / 10)
        rn = self.noise_params['Rn'] / self.z0
        g_opt = self.noise_params['gamma_opt']
        
        N = (F_target - F_min) / (4 * rn) * np.abs(1 + g_opt)**2
        
        c = g_opt / (N + 1)
        r = np.sqrt(N**2 + N * (1 - np.abs(g_opt)**2)) / (N + 1)
        
        return Circle(c, r, f"NF={F_target_db}dB")

    def get_transducer_gain(self, gamma_s: complex, gamma_l: complex) -> float:
        """Calculates G_T in dB for specific terminations."""
        gamma_in = self.S11 + (self.S12 * self.S21 * gamma_l) / (1 - self.S22 * gamma_l)
        
        term1 = (1 - np.abs(gamma_s)**2) / np.abs(1 - gamma_s * gamma_in)**2
        term2 = np.abs(self.S21)**2
        term3 = (1 - np.abs(gamma_l)**2) / np.abs(1 - self.S22 * gamma_l)**2
        
        return 10 * np.log10(term1 * term2 * term3)