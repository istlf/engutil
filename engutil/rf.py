import numpy as np
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import engutil
import math

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
    
    @property
    def Gamma_Ms(self):
        # Calculate stability factor K first to see if a match is even possible
        # K = (1 - |S11|^2 - |S22|^2 + |delta|^2) / (2 * |S12 * S21|)
        
        B1 = 1 + np.abs(self.S11)**2 - np.abs(self.S22)**2 - np.abs(self.delta)**2
        C1 = self.S11 - self.delta * np.conj(self.S22)
        
        # Cast the radicand to complex to avoid NaN if K < 1
        radicand = np.array(B1**2 - 4 * np.abs(C1)**2, dtype=complex)
        
        # Try the minus sign (usually the one for |Gamma| < 1)
        g_ms = (B1 - np.sqrt(radicand)) / (2 * C1)
        
        # If the result is outside the Smith Chart, use the plus sign
        if np.abs(g_ms) > 1:
            g_ms = (B1 + np.sqrt(radicand)) / (2 * C1)
            
        return g_ms
    
    @property
    def Gamma_ML(self):
        B2 = 1 + np.abs(self.S22)**2 - np.abs(self.S11)**2 - np.abs(self.delta)**2
        C2 = self.S22 - self.delta*np.conjugate(self.S11)
        return (B2 - np.sqrt(B2**2 - 4*np.abs(C2)**2))/(2*C2)


    @property
    def U(self):
        # Unilateral figure of merit
        # Equation 12.46 in Pozar 
        num = np.abs(self.S12)*np.abs(self.S21)*np.abs(self.S11)*np.abs(self.S22)
        den = (1 - np.abs(self.S11)**2)*(1 - np.abs(self.S22)**2)

        return num/den 
    
    @property
    def unilateral_error_limits(self):
        """
        Calculates the bounds on the ratio GT / GTU due to the unilateral assumption.
        Returns a dict with linear bounds and dB bounds.
        """
        u = self.U
        
        # Calculate linear bounds
        lower_lin = 1 / (1 + u)**2
        upper_lin = 1 / (1 - u)**2
        
        # Calculate dB bounds (Power ratio, so use 10*log10)
        # Re-using the logic from our previous conversation
        lower_db = 10 * np.log10(lower_lin)
        upper_db = 10 * np.log10(upper_lin)
        
        return {
            'lower_lin': lower_lin,
            'upper_lin': upper_lin,
            'lower_db': lower_db,
            'upper_db': upper_db,
            'range_db': (lower_db, upper_db)
        }
    @property
    def print_unilateral_report(self):
        """A helper to print the figure of merit analysis."""
        stats = self.unilateral_error_limits
        print(f"Unilateral Figure of Merit (U): {self.U:.4f}")
        print(f"Max Error Bounds: {stats['lower_db']:.2f} dB < GT/GTU < {stats['upper_db']:.2f} dB")
        
    # --- Gain Properties ---
    @property
    def max_stable_gain_db(self):
        return 10 * np.log10(np.abs(self.S21) / np.abs(self.S12))
    
    @property 
    def G_Smax(self):
        return 1/(1 - np.abs(self.S11)**2)

    @property 
    def G_Lmax(self):
        return 1/(1 - np.abs(self.S22)**2)
    
    @property
    def G0(self):
        return np.abs(self.S21)**2
    
    @property 
    def G_TUmax(self):
        # 12.42 in Pozar 
        # assumes conjugate matchin
        return self.G_Smax*self.G0*self.G_Lmax

    @property
    def G_Tmax(self):
        # 12.43 in Pozar, assumes K > 1 ie. unconditionally stable
        term1 = np.abs(self.S21)/np.abs(self.S12)
        term2 = (self.K - np.sqrt(self.K**2 - 1))   
    
        return term1*term2

    @property
    def MSG(self):
        # 12.44 in Pozar, maximum stable again - it is G_Tmax with K = 1
        return np.abs(self.S21)/np.abs(self.S12)

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

    # --- Noise Figure Circles ---

    def noise_circle(self, F_target_db: float) -> Circle:
        if not self.noise_params:
            raise ValueError("Noise parameters (Fmin_dB, Rn, gamma_opt) not provided.")
        
        F_min = 10**(self.noise_params['Fmin_dB'] / 10)
        F_target = 10**(F_target_db / 10)
        rn = self.noise_params['Rn']/self.z0


        g_opt = self.noise_params['gamma_opt']
        
        N = (F_target - F_min) / (4 * rn) * np.abs(1 + g_opt)**2
        
        c = g_opt / (N + 1)
        r = np.sqrt(N*(N + 1 - np.abs(g_opt)**2))/(N+1)
        
        return Circle(c, r, f"NF={F_target_db}dB")
    
    # --- Constant GS and GL circles --- 

    def constant_GS_circle(self, gain_db: float) -> Circle:
        """Calculates the constant GS circle for the unilateral case - see example 12.4 in pozar"""
        GSmax = self.G_Smax
        gS = (10**(gain_db/10))/GSmax

        CS = (gS * np.conjugate(self.S11))/(1 - (1 - gS) * np.abs(self.S11)**2)
        RS = (np.sqrt(1 - gS) * (1 - np.abs(self.S11)**2))/(1 - (1 - gS) * np.abs(self.S11)**2)

        return Circle(CS, RS, f"GS={gain_db}dB")
    
    def constant_GL_circle(self, gain_db: float) -> Circle:
        """Calculates the constant GL circle for the unilateral case - see example 12.4 in pozar"""
        GLmax = self.G_Lmax
        gL = (10**(gain_db/10))/GLmax

        CL = (gL * np.conjugate(self.S22))/(1 - (1 - gL) * np.abs(self.S22)**2)
        RL = (np.sqrt(1 - gL) * (1 - np.abs(self.S22)**2))/(1 - (1 - gL) * np.abs(self.S22)**2)

        return Circle(CL, RL, f"GL={gain_db}dB")

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

def to_linear(val):
    return 10**(val/10)

def reflection_2_impedance(gamma):
    return (1 + gamma)/(1-gamma)
    
def L_LP_to_BP(omega_0, Delta, L, R0):
    """Transforms a LP prototype inductor into series L and C BP components"""
    Ls = L*R0/(omega_0 * Delta)
    Cs = Delta/(omega_0 * L * R0)
    return Ls, Cs

def C_LP_to_BP(omega_0, Delta, C, R0):
    """Transforms a LP prototype capacitor into parallel L and C BP components"""
    Lp = Delta * R0/(omega_0 * C)
    Cp = C/(omega_0 * Delta * R0)
    return Lp, Cp


def parse_ads_data(file_path):
    """
    Parses an ADS exported .txt file and returns frequency and magnitude vectors.
    """
    try:
        # sep='\s+' handles one or more spaces/tabs as a delimiter
        # skiprows=1 skips the header 'freq dB(S(4,3))'
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, names=['freq', 'mag'])
        
        freq = df['freq'].values
        magnitude = df['mag'].values
        
        return freq, magnitude

    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def get_prefix(val):

    # Define SI prefixes
    prefixes = {
        -15: 'f',  # femto
        -12: 'p',  # pico
        -9: 'n',   # nano
        -6: r'\mu ', # micro (latex symbol)
        -3: 'm',   # milli
        0: '',     # units
        3: 'k',    # kilo
        6: 'M',    # mega
        9: 'G'     # giga
    }

    if val == 0:
            exp_3 = 0
            scaled_val = 0
    else:
        exponent = math.floor(math.log10(abs(val)))
        exp_3 = int(math.floor(exponent / 3) * 3)
        scaled_val = val / (10**exp_3)

    # 3. Get prefix symbol
    prefix_symbol = prefixes.get(exp_3, f"e{exp_3}")

    return prefix_symbol, exp_3

def plot_mag(f, resp, legend, title="Title", xlim=None, ylim=None, size=(14, 5), save=None, target_freqs=None):
    
    prefix_sym, scale = get_prefix(f[1])    
    f = f/(10**scale)

    engutil.init_latex()
    plt.figure(figsize=size)
    plt.plot(f, resp, label=legend)
    if target_freqs is not None:
        for i in range(len(target_freqs)):
            target_freqs[i] = target_freqs[i]/(10**scale)
            target_s21 = np.interp(target_freqs[i], f, resp)
            plt.plot(target_freqs[i], target_s21, 'ro', markersize=8)
            plt.annotate(f"({target_freqs[i]:.2f}" + prefix_sym + "Hz" + f", {target_s21:.1f} dB)", 
                    xy=(target_freqs[i], target_s21), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    color='red',
                    fontweight='bold')
    plt.xlabel("Frequency \\textit{f} / " + prefix_sym +"Hz")
    plt.ylabel("Magnitude $\\left| S21 \\right|$ / dB ")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()