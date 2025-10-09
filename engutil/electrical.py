import numpy as np

def acoustic_params(rho, c, a):
    MA1 = 8 * rho / (3 * np.pi**2 * a)
    RA1 = 0.441 * rho * c / (np.pi * a**2)
    RA2 = rho * c / (np.pi * a**2)
    CA  = 5.94 * a**3 / (rho * c**2)
    return MA1, RA1, RA2, CA
