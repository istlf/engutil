import numpy as np 


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
    mag = 20*np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H)*180/np.pi)

    return mag, phase

# def find_max(f, H):
#     max_idx = np.argmax(H)
#     f0 = f[max_idx]
#     H_max = np.abs(H[max_idx])
#     return f0, H_max, max_idx

def _interp_cross(f_arr, Z_arr, Zr, peak_idx, side):
    """Robustly find interpolated crossing frequency on one side of peak."""
    if side == "left":
        idxs = np.arange(0, peak_idx)
    else:
        idxs = np.arange(peak_idx, len(Z_arr))

    if len(idxs) < 2:
        return np.nan

    # Try to find a sign-change bracket
    sign_changes = np.where(np.diff(np.sign(Z_arr[idxs] - Zr)) != 0)[0]
    if len(sign_changes) > 0:
        # For left take last, for right take first
        sc = sign_changes[-1] if side == "left" else sign_changes[0]
        i = idxs[sc]
        j = i + 1
        return np.interp(Zr, [Z_arr[i], Z_arr[j]], [f_arr[i], f_arr[j]])

    # If no sign change, find nearest index and pick neighbour towards peak to form a bracket
    nearest = idxs[np.argmin(np.abs(Z_arr[idxs] - Zr))]
    # choose neighbour towards peak
    if side == "left":
        if nearest == peak_idx - 1:
            i, j = nearest - 1, nearest
        else:
            i, j = nearest, nearest + 1
    else:  # right
        if nearest == peak_idx:
            i, j = nearest, nearest + 1 if nearest + 1 < len(Z_arr) else (nearest - 1, nearest)
        else:
            i, j = nearest - 1, nearest

    # validate indices
    if i < 0 or j >= len(Z_arr):
        return np.nan
    return np.interp(Zr, [Z_arr[i], Z_arr[j]], [f_arr[i], f_arr[j]])


def find_resonance_q(f, ZL, Re=None, f_range=None):
    """
    Returns: f1, f2, f0, Zr, Zmax, QM, QE, QT
    ZL may be complex or already magnitudes.
    """
    f = np.asarray(f)
    ZL = np.asarray(ZL)

    # apply frequency range mask
    if f_range is not None:
        mask = (f >= f_range[0]) & (f <= f_range[1])
        if not np.any(mask):
            raise ValueError("f_range excludes all frequencies")
        f = f[mask]
        ZL = ZL[mask]

    # magnitude
    Zmag = np.abs(ZL)

    # find peak
    peak_idx = int(np.argmax(Zmag))
    f0 = float(f[peak_idx])
    Zmax = float(Zmag[peak_idx])

    # DC resistance
    if Re is None:
        # try to estimate from lowest-frequency values (first few points)
        n_est = min(5, len(Zmag))
        Re_est = np.min(Zmag[:n_est])
    else:
        Re_est = float(Re)
    if Re_est <= 0:
        raise ValueError("Re must be positive")

    Zr = np.sqrt(Re_est * Zmax)

    # find f1 (left) and f2 (right) robustly
    f1 = _interp_cross(f, Zmag, Zr, peak_idx, side="left")
    f2 = _interp_cross(f, Zmag, Zr, peak_idx, side="right")

    # compute rc and Q factors if f1,f2 valid and distinct
    if not np.isnan(f1) and not np.isnan(f2) and (f2 > f1):
        rc = Zmax / Re_est
        QM = f0 * np.sqrt(rc) / (f2 - f1)
        QE = QM / (rc - 1) if (rc - 1) != 0 else np.nan
        QT = QM / rc if rc != 0 else np.nan
    else:
        QM = QE = QT = np.nan

    return f0, f1, f2, Zr, Zmax, QM, QE, QT

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