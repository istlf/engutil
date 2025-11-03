import numpy as np 

def radiation_impedance_piston_in_baffle(a, rho=1.2, c=344):
    """
        takes a radius a (and maybe a density rho and speed of sound c) to give
        M_A1, R_A1, R_A2 and C_A
    """


    M_A1 = 8/3*rho/(np.pi**2*a)
    R_A1 = 0.441*rho*c/(np.pi*a**2)
    R_A2 = rho*c/(np.pi*a**2)
    C_A = 5.94*a**3/(rho*c**2)

    return M_A1, R_A1, R_A2, C_A