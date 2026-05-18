import numpy as np
import engutil
import matplotlib.pyplot as plt

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
    
    # --- Initial Conditions (t=0, z=0) ---

    def v0_even(self, RS, VS):
        r"""
        v(t=0, z=0)_{even} = \frac{Z_{0,even}}{R_S + Z_{0,even}} V_S
        """
        ze = self.Z0_even
        return (ze / (RS + ze)) * VS

    def i0_even(self, RS, VS):
        r"""
        i(t=0, z=0)_{even} = \frac{v(t=0, z=0)_{even}}{Z_{0,even}}
        """
        return self.v0_even(RS, VS) / self.Z0_even
    
    def v_inf_even(self, RS, RT, VS):
        r"""
        v(t=\infty)_{rising} = \frac{R_t}{R_S + R_t} V_S
        """
        return (RT / (RS + RT)) * VS

    def i_inf_even(self, RS, RT, VS):
        r"""
        i(t=\infty)_{rising} = \frac{V_S}{R_S + R_t}
        """
        return VS / (RS + RT)

    def v0_odd(self, RS, VS):
        r"""
        v(t=0, z=0)_{odd} = \frac{Z_{0,odd}}{R_S + Z_{0,odd}} V_S
        """
        zo = self.Z0_odd
        return (zo / (RS + zo)) * VS

    def i0_odd(self, RS, VS):
        r"""
        i(t=0, z=0)_{odd} = \frac{v(t=0, z=0)_{odd}}{Z_{0,odd}}
        """
        return self.v0_odd(RS, VS) / self.Z0_odd
    
    # --- Reflection Coefficients ---

    def gamma_source_even(self, RS):
        r"""
        \Gamma(z=0)_{even} = \frac{R_S - Z_{0,even}}{R_S + Z_{0,even}}
        """
        ze = self.Z0_even
        return (RS - ze) / (RS + ze)

    def gamma_load_even(self, RT):
        r"""
        \Gamma(z=l)_{even} = \frac{R_T - Z_{0,even}}{R_T + Z_{0,even}}
        """
        ze = self.Z0_even
        return (RT - ze) / (RT + ze)

    def gamma_source_odd(self, RS):
        r"""
        \Gamma(z=0)_{odd} = \frac{R_S - Z_{0,odd}}{R_S + Z_{0,odd}}
        """
        zo = self.Z0_odd
        return (RS - zo) / (RS + zo)

    def gamma_load_odd(self, RT):
        r"""
        \Gamma(z=l)_{odd} = \frac{R_T - Z_{0,odd}}{R_T + Z_{0,odd}}
        """
        zo = self.Z0_odd
        return (RT - zo) / (RT + zo)
    
    # --- Steady State Values (t=inf) ---

    def v_inf_rising(self, RS, RT, VS):
        r"""
        v(t=\infty)_{rising} = \frac{R_t}{R_S + R_t} V_S
        """
        return (RT / (RS + RT)) * VS

    def i_inf_rising(self, RS, RT, VS):
        r"""
        i(t=\infty)_{rising} = \frac{V_S}{R_S + R_t}
        """
        return VS / (RS + RT)

    def v_inf_falling(self):
        r"""
        v(t=\infty)_{falling} = 0.000 V
        """
        return 0.0

    def i_inf_falling(self):
        r"""
        i(t=\infty)_{falling} = 0.00 mA
        """
        return 0.0
    
    # --- Propagation Delay ---

    def td_even(self, length):
        r"""
        t_{d,even} = \frac{l}{v_{p,even}}
        """
        return length / self.vp_even

    def td_odd(self, length):
        r"""
        t_{d,odd} = \frac{l}{v_{p,odd}}
        """
        return length / self.vp_odd

    def plot_slem(self, VS, RT, RS, length, mode="even", edge="rising", start=1.0, num_bounces=5, rise_time=0.1):
        """
        Unified plotting function for transmission line transients.
        
        Args:
            VS, RT, RS: Source Voltage, Termination Resistance, Source Resistance.
            length: Physical length of line [m].
            mode: "even" or "odd".
            edge: "rising", "falling", or "both" (differential).
            start: Time the transition starts [ns].
            num_bounces: Number of wave arrivals to simulate.
            rise_time: Time for the signal to transition [ns].
        """
        # 1. Parameter Selection
        if mode == "even":
            td = self.td_even(length) * 1e9
            v_inc = self.v0_even(RS, VS)
            gamma_l = self.gamma_load_even(RT)
            gamma_s = self.gamma_source_even(RS)
        else:
            td = self.td_odd(length) * 1e9
            v_inc = self.v0_odd(RS, VS)
            gamma_l = self.gamma_load_odd(RT)
            gamma_s = self.gamma_source_odd(RS)
            
        v_ss = (RT / (RS + RT)) * VS  # Steady state voltage

        # 2. Simulation Setup
        t_max = start + (num_bounces + 1) * td
        t_axis = np.linspace(0, t_max, 2500)
        
        # We calculate a 'base_rising' signal (0 to Vss)
        # All other cases (falling, differential) are derived from this.
        z0_base = np.zeros_like(t_axis)
        zl_base = np.zeros_like(t_axis)

        def ramp(t, t_arr, tr): return np.clip((t - t_arr) / tr, 0, 1)

        curr_amp = v_inc
        for i in range(num_bounces):
            t_arr = start + i * td
            if i == 0:
                z0_base += curr_amp * ramp(t_axis, t_arr, rise_time)
            elif i % 2 == 1: # Hits load (z=l)
                zl_base += curr_amp * (1 + gamma_l) * ramp(t_axis, t_arr, rise_time)
                curr_amp *= gamma_l
            else: # Returns to source (z=0)
                z0_base += curr_amp * (1 + gamma_s) * ramp(t_axis, t_arr, rise_time)
                curr_amp *= gamma_s

        # 3. Handle Edge Types
        plt.figure(figsize=(10, 6))
        
        if edge == "rising":
            plt.plot(t_axis, z0_base, 'k-', label='$v(z=0)$')
            plt.plot(t_axis, zl_base, 'k--', label='$v(z=l)$')
            title = f"Rising Edge ({mode.capitalize()} Mode)"
            
        elif edge == "falling":
            plt.plot(t_axis, v_ss - z0_base, 'k-', label='$v(z=0)$')
            plt.plot(t_axis, v_ss - zl_base, 'k--', label='$v(z=l)$')
            title = f"Falling Edge ({mode.capitalize()} Mode)"
            
        elif edge == "both":
            # Rising pair
            plt.plot(t_axis, z0_base, 'k-', label='$v_{rise}(z=0)$')
            plt.plot(t_axis, zl_base, 'k--', label='$v_{rise}(z=l)$')
            # Falling pair
            plt.plot(t_axis, v_ss - z0_base, 'r-', label='$v_{fall}(z=0)$')
            plt.plot(t_axis, v_ss - zl_base, 'r--', label='$v_{fall}(z=l)$')
            title = f"Both rising and falling edge ({mode.capitalize()} Mode)"

        # 4. Styling
        plt.title(title)
        plt.xlabel('Time [ns]')
        plt.ylabel('Voltage [V]')
        plt.xlim(0, t_max)
        plt.ylim(-0.1, v_ss + 0.1)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right', fontsize='small')
        plt.show()

    