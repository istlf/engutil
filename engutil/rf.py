import numpy as np
import engutil
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

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
    def Sopt(self): return self.noise_params["gamma_opt"]
    @property
    def Fmin_dB(self): return self.noise_params["Fmin_dB"]
    @property
    def Rn(self): return self.noise_params["Rn"]
    
    @property
    def delta(self):
        return (self.S11 * self.S22) - (self.S12 * self.S21)
        
    @property
    def K(self):
        num = 1 - np.abs(self.S11)**2 - np.abs(self.S22)**2 + np.abs(self.delta)**2
        den = 2 * np.abs(self.S12 * self.S21)
        return num / den
    
    @property
    def is_unconditionally_stable(self):
        # returns true if K > 1 and delta < 1 
        return (self.K > 1) and (np.abs(self.delta) < 1)
    def get_z_in(self, gamma_l):
        gin = self.gamma_in(gamma_l)
        return self.z0 * (1 + gin) / (1 - gin)

    def get_z_out(self, gamma_s):
        gout = self.gamma_out(gamma_s)
        return self.z0 * (1 + gout) / (1 - gout)
    
    def get_z_in(self, gamma_l):
        gin = self.gamma_in(gamma_l)
        return self.z0 * (1 + gin) / (1 - gin)

    def get_z_out(self, gamma_s):
        gout = self.gamma_out(gamma_s)
        return self.z0 * (1 + gout) / (1 - gout)

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

    def Gamma_Ms(self, latex=False):
        # Calculate intermediate values
        # Calculate stability factor K first to see if a match is even possible
        # K = (1 - |S11|^2 - |S22|^2 + |delta|^2) / (2 * |S12 * S21|)
        
        B1 = 1 + np.abs(self.S11)**2 - np.abs(self.S22)**2 - np.abs(self.delta)**2
        C1 = self.S11 - self.delta * np.conj(self.S22)
        
        radicand = np.array(B1**2 - 4 * np.abs(C1)**2, dtype=complex)
        
        # Determine sign for stability
        g_ms = (B1 - np.sqrt(radicand)) / (2 * C1)
        sign_used = "-"
        if np.abs(g_ms) > 1:
            g_ms = (B1 + np.sqrt(radicand)) / (2 * C1)
            sign_used = "+"
            
        if latex:
            def cfmt(z):
                r, phi = engutil.to_polar(z)
                return rf"{r:.4f}\angle {phi:.2f}^\circ"
            
            # intermediate LaTeX steps
            latex_str = rf"""
\begin{{aligned}}
B_1 &= 1 + |S_{{11}}|^2 - |S_{{22}}|^2 - |\Delta|^2 = {B1:.4f} \\
C_1 &= S_{{11}} - \Delta S_{{22}}^* = {cfmt(C1)} \\
\Gamma_{{Ms}} &= \frac{{B_1 {sign_used} \sqrt{{B_1^2 - 4|C_1|^2}}}}{{2C_1}} = {cfmt(g_ms)}
\end{{aligned}}
"""
            print(latex_str)
            return g_ms, latex_str

        return g_ms

    def Gamma_ML(self, latex=False):
        # Calculate intermediate values
        B2 = 1 + np.abs(self.S22)**2 - np.abs(self.S11)**2 - np.abs(self.delta)**2
        C2 = self.S22 - self.delta * np.conj(self.S11)
        
        radicand = np.array(B2**2 - 4 * np.abs(C2)**2, dtype=complex)
        
        # Determine sign for stability
        g_ml = (B2 - np.sqrt(radicand)) / (2 * C2)
        sign_used = "-"
        if np.abs(g_ml) > 1:
            g_ml = (B2 + np.sqrt(radicand)) / (2 * C2)
            sign_used = "+"

        if latex:
            def cfmt(z):
                r, phi = engutil.to_polar(z)
                return rf"{r:.4f}\angle {phi:.2f}^\circ"
            
            latex_str = rf"""
\begin{{aligned}}
B_2 &= 1 + |S_{{22}}|^2 - |S_{{11}}|^2 - |\Delta|^2 = {B2:.4f} \\
C_2 &= S_{{22}} - \Delta S_{{11}}^* = {cfmt(C2)} \\
\Gamma_{{ML}} &= \frac{{B_2 {sign_used} \sqrt{{B_2^2 - 4|C_2|^2}}}}{{2C_2}} = {cfmt(g_ml)}
\end{{aligned}}
"""
            print(latex_str)
            return g_ml, latex_str
            
        return g_ml
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
        upper_lin = 1 / (1 -  u)**2
        
        # Calculate dB bounds (Power ratio, so use 10*log10)
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
    
    def G_Smax(self, db=False):
        gain = 1/(1 - np.abs(self.S11)**2)

        return engutil.pow2db(gain) if db else gain 

    def G_Lmax(self, db=False):
        gain =  1/(1 - np.abs(self.S22)**2)
        return engutil.pow2db(gain) if db else gain 
    
    def G0(self, db=False):
        gain = np.abs(self.S21)**2
        return engutil.pow2db(gain) if db else gain 
    
    def G_TUmax(self, db=False):
        # 12.42 in Pozar 
        # assumes conjugate matchin
        gain = self.G_Smax()*self.G0()*self.G_Lmax()
        return engutil.pow2db(gain) if db else gain

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

    def MAG(self, db=True):
        """
        Calculates Maximum Available Gain (MAG). 
        Only defined if K > 1 and |delta| < 1.
        """
        if self.is_unconditionally_stable:
            # 12.43 in Pozar
            term1 = np.abs(self.S21) / np.abs(self.S12)
            term2 = self.K - np.sqrt(self.K**2 - 1)
            return engutil.pow2db(term1 * term2) if db else term1 * term2
        else:
            return None # Or raise an error

    @property
    def max_gain_limit(self):
        """
        A 'smart' property that returns MAG if stable, or MSG if unstable.
        This is what you would see on a datasheet 'Maximum Gain' plot.
        """
        if self.is_unconditionally_stable:
            return self.MAG
        else:
            return self.MSG
        

    # --- Constant Gain Circles (Available Gain) ---

    def available_gain_circle(self, gain_db: float) -> Circle:
        """Calculates the Ga circle for the Source plane (Gamma_S) - page 257 in Gonzales"""
        Ga =  10**(gain_db / 10)
        ga = Ga / (np.abs(self.S21)**2)
        

        
        c1 = self.S11 - self.delta * np.conj(self.S22)
        #print(f"Ga: {Ga} => ga: {ga}, c1: {engutil.to_polar(c1, latex=True)}")
        
        den = 1 + ga * (np.abs(self.S11)**2 - np.abs(self.delta)**2)
        
        ca = (ga * np.conj(c1)) / den
        
        s12s21 = np.abs(self.S12 * self.S21)

        num_r = np.sqrt(1 - 2 * self.K * s12s21 * ga + (s12s21 * ga)**2)
        ra = num_r / np.abs(den)
        #print(f"CA: {engutil.to_polar(ca, latex=True)} and rs: {engutil.to_polar(ra, latex=True)}")
        return Circle(ca, ra, f"Ga={gain_db}dB")

    # --- Noise Figure Circles ---

    def noise_circle(self, F_target_db: float) -> Circle:
        if not self.noise_params:
            raise ValueError("Noise parameters (Fmin_dB, Rn, gamma_opt) not provided.")
        
        F_min = 10**(self.noise_params['Fmin_dB'] / 10)
        F_target = 10**(F_target_db / 10)
        rn = self.noise_params['Rn']/self.z0
        print(f"F_target_db: {F_target_db} F_target_lin: {F_target} F_min_lin: {F_min} rn: {rn} ")

        g_opt = self.noise_params['gamma_opt']
        print(f"g_opt: {engutil.to_polar(g_opt,latex=True)}")
        N = (F_target - F_min) / (4 * rn) * np.abs(1 + g_opt)**2
        print(f"N: {engutil.to_polar(N, latex=True)}")
        c = g_opt / (N + 1)
        r = np.sqrt(N*(N + 1 - np.abs(g_opt)**2))/(N+1)
        print(f"c: {engutil.to_polar(c,latex=True)}, r: {engutil.to_polar(r,latex=True)}")

        return Circle(c, r, f"NF={F_target_db}dB")
    



    def unilateral_source_gain_circle(self, gain_db: float) -> Circle:
        """Calculates the Constant Gain Circle (Gs) for the source plane (Unilateral)."""
        Gs_lin = 10**(gain_db / 10)
        Gs_max = 1 / (1 - np.abs(self.S11)**2)
        
        # Normalized gain (g_s in Pozar/your image)
        gs = Gs_lin / Gs_max
        
        # Pozar 12.51a and 12.51b
        den = 1 - (1 - gs) * np.abs(self.S11)**2
        c = (gs * np.conj(self.S11)) / den
        r = (np.sqrt(1 - gs) * (1 - np.abs(self.S11)**2)) / den
        
        return Circle(c, r, f"Gs={gain_db}dB")

    def unilateral_load_gain_circle(self, gain_db: float) -> Circle:
        """Calculates the Constant Gain Circle (Gl) for the load plane (Unilateral)."""
        Gl_lin = 10**(gain_db / 10)
        Gl_max = 1 / (1 - np.abs(self.S22)**2)
        
        # Normalized gain (g_l in Pozar/your image)
        gl = Gl_lin / Gl_max
        
        # Pozar 12.52a and 12.52b
        den = 1 - (1 - gl) * np.abs(self.S22)**2
        c = (gl * np.conj(self.S22)) / den
        r = (np.sqrt(1 - gl) * (1 - np.abs(self.S22)**2)) / den
        
        return Circle(c, r, f"Gl={gain_db}dB")




    def vswr_circle(self, center_gamma: complex, vswr: float) -> Circle:
        """
        Generates a circle of constant mismatch (VSWR) around a target reflection point.
        Useful for 'mismatching on purpose' for stability.
        """
        rho = (vswr - 1) / (vswr + 1)
        print(f"gamma_b: {rho}")
        # For a circle around a non-zero point in the Gamma plane:
        num_c = center_gamma * (1 - rho**2)
        den_c = 1 - (np.abs(center_gamma)**2 * rho**2)
        
        c = num_c / den_c
        r = (rho * (1 - np.abs(center_gamma)**2)) / den_c
        print(f"C_VSWR: {engutil.to_polar(c, latex=True)} and r: {r}")
        return Circle(c, r, f"VSWR={vswr} around {to_polar(center_gamma)}")

    def input_vswr_50ohm(self, gamma_s, gamma_l):
        """
        Calculates the VSWR seen at the 50 Ohm input port.
        This assumes your input matching network transforms 50 Ohms -> gamma_s.
        """
        g_in = self.gamma_in(gamma_l)
        # This calculates the mismatch between what the matching network 
        # expects (gamma_s) and what the transistor actually provides (g_in)
        # Then it transforms that back to the 50 Ohm reference.
        gamma_ref = np.abs((g_in - np.conj(gamma_s)) / (1 - g_in * gamma_s))
        return (1 + gamma_ref) / (1 - gamma_ref)

    def gamma_in(self, gamma_l):
        return self.S11 + (self.S12 * self.S21 * gamma_l) / (1 - self.S22 * gamma_l)

    def gamma_out(self, gamma_s):
        return self.S22 + (self.S12 * self.S21 * gamma_s) / (1 - self.S11 * gamma_s)

    # def G_T(self, gamma_s, gamma_l, db=False): 
    #     # Gonzales 3.2.1 
    #     g_in = self.gamma_in(gamma_l)
        
    #     term_s = (1 - np.abs(gamma_s)**2) / np.abs(1 - g_in * gamma_s)**2
    #     term_0 = np.abs(self.S21)**2
    #     term_l = (1 - np.abs(gamma_l)**2) / np.abs(1 - self.S22 * gamma_l)**2
        
    #     gain = term_s * term_0 * term_l
    #     return engutil.pow2db(gain) if db else gain

    def G_T(self, gamma_s, gamma_l, db=False, latex=False):
        # Gonzales 3.2.1 
        g_in = self.gamma_in(gamma_l)
        print(f"g_inds: {engutil.to_polar(g_in, latex=True)}")
        term_s = (1 - np.abs(gamma_s)**2) / np.abs(1 - g_in * gamma_s)**2
        term_0 = np.abs(self.S21)**2
        term_l = (1 - np.abs(gamma_l)**2) / np.abs(1 - self.S22 * gamma_l)**2
        
        gain = term_s * term_0 * term_l

        if latex:
            def cfmt(z):
                r, phi = engutil.to_polar(z)
                return rf"{r:.4f}\angle {phi:.2f}^\circ"

            latex_str = rf"""
    G_T = 
    \frac{{1 - |{cfmt(gamma_s)}|^2}}{{|1 - ({cfmt(g_in)})({cfmt(gamma_s)})|^2}}
    \cdot |{cfmt(self.S21)}|^2
    \cdot \frac{{1 - |{cfmt(gamma_l)}|^2}}{{|1 - ({cfmt(self.S22)})({cfmt(gamma_l)})|^2}}
    = {gain:.4f}
    """
            if db:
                latex_str += rf" = {engutil.pow2db(gain):.2f}\,\text{{dB}}"
            
            print(latex_str)
            return latex_str

        return engutil.pow2db(gain) if db else gain

    def G_A(self, gamma_s, db=False):
        """
        Calculates Available Power Gain (G_A).
        This assumes conjugate matching on the output (gl = gout*).
        Gzonales 3.2.3
        """
        g_out = self.gamma_out(gamma_s)
        term1 = (1 - np.abs(gamma_s)**2) / np.abs(1 - self.S11 * gamma_s)**2
        term2 = np.abs(self.S21)**2
        term3 = 1 / (1 - np.abs(g_out)**2)
        
        gain = term1 * term2 * term3
        return engutil.pow2db(gain) if db else gain

    def G_P(self, gamma_l, db=False):
        """
        Calculates Operating Power Gain (G_P).
        This assumes conjugate matching on the input (gs = gin*).
        Gonzales 3.2.4
        """
        g_in = self.gamma_in(gamma_l)
        term1 = 1 / (1 - np.abs(g_in)**2)
        term2 = np.abs(self.S21)**2
        term3 = (1 - np.abs(gamma_l)**2) / np.abs(1 - self.S22 * gamma_l)**2
        
        gain = term1 * term2 * term3
        return engutil.pow2db(gain) if db else gain

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

import numpy as np

def reflection_2_impedance(gamma, z0=50):
    """
    Convert a complex reflection coefficient (Gamma) to impedance.

    Parameters
    ----------
    gamma : complex or array_like
        The complex reflection coefficient.
    z0 : float or complex, optional
        The characteristic impedance of the system. 
        Set to 1.0 to return normalized impedance (z). 
        Defaults to 50.

    Returns
    -------
    z : complex or ndarray
        The complex impedance (Z). Returns np.inf if gamma is 1.0.

    Examples
    --------
    >>> reflection_2_impedance(0)
    (50+0j)
    >>> reflection_2_impedance(0.33333333, z0=50) # Gamma for 100 Ohm
    (99.99999925+0j)
    """
    gamma = np.asanyarray(gamma, dtype=complex)
    
    # Use numpy.errstate to handle division by zero gracefully (at gamma=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = z0 * (1 + gamma) / (1 - gamma)
    
    # If input was a scalar, return a scalar for better usability
    return z.item() if z.ndim == 0 else z

def impedance_2_reflection(z, z0=50):
    """
    Convert a complex impedance (Z) to a reflection coefficient (Gamma).

    Parameters
    ----------
    z : complex or array_like
        The complex impedance.
    z0 : float or complex, optional
        The characteristic impedance of the system.
        If z is already normalized, set z0=1.0.
        Defaults to 50.

    Returns
    -------
    gamma : complex or ndarray
        The complex reflection coefficient.

    Examples
    --------
    >>> impedance_2_reflection(50, z0=50)
    0j
    >>> impedance_2_reflection(0, z0=50) # Short circuit
    (-1+0j)
    """
    z = np.asanyarray(z, dtype=complex)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = (z - z0) / (z + z0)
        
    return gamma.item() if gamma.ndim == 0 else gamma
def to_cartesian(polar_tuple):
    """
    Converts a (magnitude, angle_in_degrees) tuple into a complex number (a + jb).
    """
    mag, angle_deg = polar_tuple
    return mag * np.exp(1j * np.deg2rad(angle_deg))

def to_polar(complex_val, latex=False, precision=3):
    """
    Converts a complex number (a + jb) into:
      - (magnitude, angle_in_degrees) tuple (default), or
      - LaTeX string \\polar{mag}{angle} if latex=True
    """
    mag = np.abs(complex_val)
    angle_deg = np.rad2deg(np.angle(complex_val))

    if latex:
        return f"\\polar{{{mag:.{precision}f}}}{{{angle_deg:.{precision}f}}}"
    return (mag, angle_deg)

def gamma_to_vswr(gamma):
    mag = np.abs(gamma)
    return (1 + mag) / (1 - mag)
    
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


def design_coupled_line_filter(g_factors, z0=50.0, f0=1413.5e6, bw=40e6):
    """
    Calculates Even and Odd mode impedances for a coupled-line bandpass filter.
    Based on Pozar's Microwave Engineering (Section 8.8).
    
    Parameters:
    -----------
    g_factors : np.array
        The prototype g-values including g0 (source) and g_N+1 (load).
        For an Nth order filter, len(g_factors) should be N + 2.
    z0 : float
        Characteristic impedance (usually 50 Ohms).
    f0 : float
        Center frequency in Hz.
    bw : float
        Bandwidth in Hz.
        
    Returns:
    --------
    list of dicts
        A list where each element is a dictionary containing Z0e and Z0o 
        for that specific section.

    Example usage:

#Order 5
g0 = 1
g1 = 1.7058
g2 = 1.2296
g3 = 2.5408
g4 = 1.2296
g5 = 1.7058
g6 = 1

g = np.array([g0, g1, g2, g3, g4, g5, g6])

results = engutil.rf.design_coupled_line_filter(
    g_factors=g, 
    z0=50, 
    f0=1413.5e6, 
    bw=40e6
)
# 1431.5000
Z0J1 = np.sqrt(np.pi*delta/(2*g[1]))
Z0J2 = np.pi*delta/(2*np.sqrt(g[1]*g[2]))
Z0J3 = np.pi*delta/(2*np.sqrt(g[2]*g[3]))
print(f"Z0J1: {Z0J1}")
print(f"Z0J2: {Z0J2}")
print(f"Z0J3: {Z0J3}")
Z0e1 = Z0*(1 + Z0J1 + Z0J1**2)
Z0o1 =  Z0*(1 - Z0J1 + Z0J1**2)
print(f"Z0e1: {Z0e1} and Z0o1: {Z0o1}")
print(g[1])
print(f"{'Section':<8} | {'Z0e (Ω)':<10} | {'Z0o (Ω)':<10}")
print("-" * 35)
for s in results:
    print(f"{s['section']:<8} & {s['z0e']:<10.5f} & {s['z0o']:<10.5f} & {s['jz0']:.5f} \\\\")

for s in results:
    print(f"Ze{s['section']} = {s['z0e']}")
    print(f"Zo{s['section']} = {s['z0o']}")

print(g[0])
# 8.131 for coupled 
# shorted stubs for bandpass


    """
    # 1. Basic parameters
    N = len(g_factors) - 2  # Order of the filter
    delta = bw / f0         # Fractional bandwidth
    
    # 2. Calculate J-inverters (normalized: Z0*Jn)
    # There are N+1 sections for an Nth order filter
    jz0 = np.zeros(N + 1)
    
    # First section (n=1)
    jz0[0] = np.sqrt((np.pi * delta) / (2 * g_factors[0] * g_factors[1]))
    print(f"g0: {g_factors[0]}")
    # Intermediate sections (n=2 to N)
    for n in range(1, N):
        # Indexing logic: g_factors[n] is g_i, g_factors[n+1] is g_i+1
        jz0[n] = (np.pi * delta) / (2 * np.sqrt(g_factors[n] * g_factors[n+1]))
    
    # Last section (n=N+1)
    jz0[N] = np.sqrt((np.pi * delta) / (2 * g_factors[N] * g_factors[N+1]))
    
    # 3. Calculate Even and Odd mode impedances
    sections = []
    for i, val in enumerate(jz0):
        # Equations from Pozar:
        # Z0e = Z0 * (1 + JZ0 + JZ0^2)
        # Z0o = Z0 * (1 - JZ0 + JZ0^2)
        z0e = z0 * (1 + val + val**2)
        z0o = z0 * (1 - val + val**2)
        
        sections.append({
            "section": i + 1,
            "z0e": z0e,
            "z0o": z0o,
            "jz0": val
        })
        
    return sections

def design_coupled_stub_resonator(g_factors, z0=50.0, f0=1413.5e6, bw=40e6):
    """
    Eq. 8.131 

    Example usage:


#Order 5
g0 = 1
g1 = 1.7058
g2 = 1.2296
g3 = 2.5408
g4 = 1.2296
g5 = 1.7058
g6 = 1

g = np.array([g1, g2, g3, g4, g5, g6])

results = engutil.rf.design_coupled_stub_resonator(
    g_factors=g, 
    z0=50, 
    f0=1413.5e6, 
    bw=40e6
)

for s in results:
    print(f"{s["section"]:<10} & {s["g"]:<10.5f} & {s["z0"]:<10.5f}\\\\" )

for s in results:
    print(f"Z0{s["section"]} = {s["z0"]:<10}")

print(np.pi*50*delta/(4*1.7058))

    """

    delta = bw / f0  
    sections = []
    Z0n = np.zeros_like(g_factors)
    for i, gn in enumerate(g_factors):
        Z0n[i] = np.pi*z0*delta/(4*g_factors[i])
        sections.append({
            "section": i+1,
            "g": gn,
            "z0": Z0n[i]
        })
    return sections

   
    
    # sections = []
    # for i, val in enumerate(jz0):
    #     # Equations from Pozar:
    #     # Z0e = Z0 * (1 + JZ0 + JZ0^2)
    #     # Z0o = Z0 * (1 - JZ0 + JZ0^2)
    #     z0e = z0 * (1 + val + val**2)
    #     z0o = z0 * (1 - val + val**2)
        
    #     sections.append({
    #         "section": i + 1,
    #         "z0e": z0e,
    #         "z0o": z0o,
    #         "jz0": val
    #     })
        
    # return sections




def transform_series_element(gk, z0, omega0, delta):
    """
    Transforms a prototype series inductor (gk) into a series L-C tank.
    Pozar Eq 8.74a, 8.74b
    """
    L_series = (gk * z0) / (omega0 * delta)
    C_series = delta / (gk * z0 * omega0)
    return L_series, C_series

def transform_shunt_element(gk, z0, omega0, delta):
    """
    Transforms a prototype shunt capacitor (gk) into a parallel L-C tank.
    Pozar Eq 8.74c, 8.74d
    """
    L_parallel = (delta * z0) / (gk * omega0)
    C_parallel = gk / (delta * z0 * omega0)
    return L_parallel, C_parallel

# def design_lumped_bandpass(g_factors, z0=50, f0=1413.5e6, bw=40e6):
#     """
#     Iterates through g-factors and calculates all L and C values.
#     Assumes g1 is a series element.
#     """
#     omega0 = 2 * np.pi * f0
#     delta = bw / f0
    
#     # We ignore g0 and g_last for the LC components (they are R_source/R_load)
#     elements = g_factors[1:-1]
#     results = []

#     for i, gk in enumerate(elements):
#         idx = i + 1
#         if idx % 2 != 0:
#             # Odd index: Series L-C
#             L, C = transform_series_element(gk, z0, omega0, delta)
#             results.append({"type": "Series", "index": idx, "L": L, "C": C})
#         else:
#             # Even index: Parallel L-C
#             L, C = transform_shunt_element(gk, z0, omega0, delta)
#             results.append({"type": "Shunt", "index": idx, "L": L, "C": C})
            
#     return results


import numpy as np

def transform_to_series_lc(gk, z0, omega0, delta):
    """Lp prototype inductor -> Series L-C tank"""
    L = (gk * z0) / (omega0 * delta)
    C = delta / (gk * z0 * omega0)
    return L, C

def transform_to_shunt_lc(gk, z0, omega0, delta):
    """Lp prototype capacitor -> Shunt (parallel) L-C tank"""
    L = (delta * z0) / (gk * omega0)
    C = gk / (delta * z0 * omega0)
    return L, C

def design_lumped_bandpass(g_factors, z0=50, f0=1413.5e6, bw=40e6, first_element_type='series'):
    """
    first_element_type: 'series' (Series LC first) or 'shunt' (Parallel LC first)
    """
    omega0 = 2 * np.pi * f0
    delta = bw / f0
    
    # Internal g-factors (skipping g0 and g_last)
    elements = g_factors[1:-1]
    results = []

    for i, gk in enumerate(elements):
        idx = i + 1
        
        # Determine if this specific index should be series or shunt
        # If series-first: odd indices are series, even are shunt
        # If shunt-first: odd indices are shunt, even are series
        if first_element_type.lower() == 'series':
            is_series = (idx % 2 != 0)
        else:
            is_series = (idx % 2 == 0)

        if is_series:
            L, C = transform_to_series_lc(gk, z0, omega0, delta)
            results.append({"type": "Series", "index": idx, "L": L, "C": C})
        else:
            L, C = transform_to_shunt_lc(gk, z0, omega0, delta)
            results.append({"type": "Shunt", "index": idx, "L": L, "C": C})
            
    return results

def _to_polar_latex(self, z, precision=3):
        """Helper to convert complex number to LaTeX polar form: r ∠ θ°"""
        r = np.abs(z)
        theta = np.angle(z, deg=True)
        return f"{r:.{precision}f} \\angle {theta:.{precision}f}^\\circ"

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





def conversion_gain_corrected(
    G_minus1, # G matrix
    G0, # g matrix 
    Y_w1, # IF admittance
    Y_w1_w0, # RF emb admittance
    Yin_w1_w0,
    Z_w1, # IF emb impedance 
    Z_w1_w0, # RF emb impedance
    Rs # 13 
):
    """_summary_

    Parameters
    ----------
    G_minus1 : _type_
        _description_
    Z_w1 : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Example usage: 
        R_s = 13 
        Y_in = G[0] - G[1]*G[1]/(G[0] + Y_emb_IF)
        G_cnv_corrected = conversion_gain_corrected(G[1], G[0], Y_emb_IF, Y_emb_RF, Y_in, Z_emb_IF, Z_emb_RF, R_s)
        G_cnv = complex(G_cnv_corrected).real
        G_cnv_db = engutil.pow2db(G_cnv)

    """
    # Eq. 3.73 in nonlin analysis 
    G_cnv = (
        abs(G_minus1 / (G0 + Y_w1))**2
        * abs(1 / (Y_w1_w0 + Yin_w1_w0))**2
        * 4 * Y_w1_w0.real * Y_w1.real
    )
    #print(f"G_cnv_uncorr: {G_cnv:.5f} => {engutil.pow2db(G_cnv):.5f}")
    # Eq. 3.74
    G_cnv_corrected = (
        (Z_w1_w0.real - Rs) / Z_w1_w0.real
        * G_cnv
        * (Z_w1.real - Rs) / Z_w1.real
    )

    return G_cnv_corrected