import numpy as np 
import math 
import inspect
import sympy as sp 


def load_complex_csv(path, single_line=False):
    """
    Load CSV file containing complex numbers (with 'i' as imaginary unit) from matlab exports
    ie. matlab will export complex numbers as 123+-456i but we want to convert that to 123-456j
    """
    with open(path) as fn:
        if single_line:
            line = fn.readline().strip()
            data = [
                complex(entry.replace("i", "j").replace("+-", "-"))
                for entry in line.split(",")
            ]
        else:
            data = [
                complex(line.strip().replace("i", "j").replace("+-", "-"))
                for line in fn
            ]
    return np.array(data)


def round_to_E(value, series='E12'):
    E_values = {
        'E6':  [1.0, 1.5, 2.2, 3.3, 4.7, 6.8],
        'E12': [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
        'E24': [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
    }[series]
    
    decade = 10 ** np.floor(np.log10(value))
    norm = value / decade
    nearest = min(E_values, key=lambda x: abs(x - norm))
    return nearest * decade




def open_csv(file_name):
    with open(file_name) as fn:
        H = [
            complex(line.strip().replace("i", "j").replace("+-", "-"))
            for line in fn
        ]
    return np.array(H)

def read_bode_csv(file_name):
    data = open_csv(file_name)
    mag = 20*np.log10(np.abs(data))
    phase = np.unwrap(np.angle(data))*180/np.pi
    return mag, phase

def tf_to_magphase(H):
    mag_db = 20*np.log10(np.abs(H))
    mag_lin = np.abs(H)
    phase = np.unwrap(np.angle(H)*180/np.pi)

    return mag_db, phase, mag_lin

def find_f1_f2(f, ZL, R_E, f_range=None):
    """
    Find f1 and f2 where |ZL| crosses sqrt(Re * Zmax),
    optionally within a frequency range.
    """
    # Apply optional range
    if f_range is not None:
        mask = (f >= f_range[0]) & (f <= f_range[1])
        f = f[mask]
        ZL = ZL[mask]

    # Find resonance peak
    f0, Zmax, Zmax_idx = find_max(f, ZL, f_range=f_range)

    # DC resistance
    # Re_1 = np.abs(ZL[0]) if Re is None else Re
    # Zr = np.sqrt(np.abs(Re_1) * Zmax)
    Zr = np.sqrt(R_E*Zmax)

    # Split data
    left_f = f[:Zmax_idx]
    left_Z = np.abs(ZL[:Zmax_idx])
    right_f = f[Zmax_idx:]
    right_Z = np.abs(ZL[Zmax_idx:])

    # Left crossing
    cross_left = np.where(np.diff(np.sign(left_Z - Zr)))[0]
    if len(cross_left) > 0:
        i = cross_left[-1]
        f1 = np.interp(Zr, [left_Z[i], left_Z[i+1]], [left_f[i], left_f[i+1]])
    else:
        f1 = np.nan

    # Right crossing
    cross_right = np.where(np.diff(np.sign(right_Z - Zr)))[0]
    if len(cross_right) > 0:
        i = cross_right[0]
        f2 = np.interp(Zr, [right_Z[i], right_Z[i+1]], [right_f[i], right_f[i+1]])
    else:
        f2 = np.nan

    return f1, f2, Zr
    
def find_max(f, H, f_range=None):
    """
    Find the maximum magnitude in H, optionally within a frequency range.

    Parameters
    ----------
    f : array_like
        Frequency array.
    H : array_like
        Response array (complex or real).
    f_range : tuple or list, optional
        (f_min, f_max) range to search within.

    Returns
    -------
    f0 : float
        Frequency of maximum.
    H_max : float
        Maximum magnitude.
    max_idx : int
        Index of maximum in the full array.
    """
    if f_range is not None:
        mask = (f >= f_range[0]) & (f <= f_range[1])
        f_sub, H_sub = f[mask], H[mask]
        rel_idx = np.argmax(np.abs(H_sub))
        max_idx = np.where(mask)[0][rel_idx]
    else:
        max_idx = np.argmax(np.abs(H))

    f0 = f[max_idx]
    H_max = np.abs(H[max_idx])
    return f0, H_max, max_idx


# def pprint(*args):
#     frame = inspect.currentframe().f_back
#     for var in args:
#         for name, val in frame.f_locals.items():
#             if var is val:
#                 print(f"{name:<15}: {val:.5e}")
#                 break

def pprint(*args, eng=True):
    frame = inspect.currentframe().f_back
    for var in args:
        for name, val in frame.f_locals.items():
            if var is val:
                # Convert sympy numbers to float if possible
                if isinstance(val, (sp.Float, sp.Integer, sp.Rational)):
                    val = float(val)

                if isinstance(val, (int, float)) and val != 0 and eng:
                    exp = int(math.floor(math.log10(abs(val)) / 3) * 3)
                    eng_val = val / (10 ** exp)
                    suffix = f"e{exp:+d}" if exp != 0 else ""
                    print(f"{name:<15}: {eng_val:>12.6f}{suffix}")
                else:
                    print(f"{name:<15}: {val}")
                break
