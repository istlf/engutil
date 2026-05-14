import matplotlib.pyplot as plt
import math
import engutil
import pandas as pd
import numpy as np

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

def _get_prefix(val):

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
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 30,
        "axes.labelsize": 30,
        "axes.titlesize": 32
    })

    prefix_sym, scale = _get_prefix(f[1])    
    f = f/(10**scale)

    plt.figure(figsize=size)
    plt.plot(f, resp, label=legend)
    if target_freqs is not None:
        for i in range(len(target_freqs)):
            target_freqs[i] = target_freqs[i]/(10**scale)
            target_s21 = np.interp(target_freqs[i], f, resp)
            plt.plot(target_freqs[i], target_s21, 'ro', markersize=8)
            plt.annotate(f"({target_freqs[i]:.4f}" + prefix_sym + "Hz" + f", {target_s21:.1f} dB)", 
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


def get_bandpass_stats(freq, mag, is_db=False):
    """
    Calculates f0, fL, and fH for a band-pass filter.
    
    Parameters:
    freq (array): Array of frequency values
    mag (array): Array of magnitude values (Linear or dB)
    is_db (bool): Set to True if mag is already in dB. 
                  If False, assumes linear magnitude.
    
    Returns:
    tuple: (f0, fL, fH)
    """
    # 1. Convert to dB if it's linear to make -3dB calculation standard
    if not is_db:
        # Avoid log of zero
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    else:
        mag_db = mag

    # 2. Find the peak (f0)
    peak_idx = np.argmax(mag_db)
    f0 = freq[peak_idx]
    max_mag = mag_db[peak_idx]
    target_mag = max_mag - 3.01  # -3dB point
    
    # 3. Split into left and right halves to find cutoffs
    left_mag = mag_db[:peak_idx+1]
    left_freq = freq[:peak_idx+1]
    
    right_mag = mag_db[peak_idx:]
    right_freq = freq[peak_idx:]
    
    # 4. Interpolate to find fL (Lower Cutoff)
    # np.interp requires the 'x' array (magnitudes) to be increasing
    # So we use the left side
    try:
        fL = np.interp(target_mag, left_mag, left_freq)
    except:
        fL = None # Target not reached on the left
        
    # 5. Interpolate to find fH (Upper Cutoff)
    # For the right side, magnitude is decreasing, so we flip arrays to make it increasing
    try:
        fH = np.interp(target_mag, right_mag[::-1], right_freq[::-1])
    except:
        fH = None # Target not reached on the right

    return f0, fL, fH

def get_filter_bounds(freq, mag, is_db=False):
    """
    Finds -3dB frequencies first, then calculates f0.
    
    Parameters:
    freq (array): Frequency values.
    mag (array): Magnitude values.
    is_db (bool): If True, treats mag as dB. If False, converts linear to dB.
    
    Returns:
    dict: {fL, fH, f0_geom, f0_arith, bandwidth}
    """
    # 1. Ensure we are working in dB
    if not is_db:
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    else:
        mag_db = mag

    # 2. Find the reference point (the maximum magnitude)
    # Even in non-ideal filters, -3dB is relative to the peak of the passband
    max_idx = np.argmax(mag_db)
    ref_mag = mag_db[max_idx]
    target_mag = ref_mag - 3.01
    
    # 3. Find fL (Lower Cutoff)
    # Search from the start of the array up to the peak
    left_mag = mag_db[:max_idx+1]
    left_freq = freq[:max_idx+1]
    # We use interp, but we must ensure the 'x' array is increasing.
    # On the left slope, magnitude is increasing, so this works:
    fL = np.interp(target_mag, left_mag, left_freq)
    
    # 4. Find fH (Upper Cutoff)
    # Search from the peak to the end of the array
    right_mag = mag_db[max_idx:]
    right_freq = freq[max_idx:]
    # On the right slope, magnitude is decreasing. 
    # To use np.interp, we flip the arrays so magnitude is increasing.
    fH = np.interp(target_mag, right_mag[::-1], right_freq[::-1])
    
    # 5. Calculate Center Frequency (f0)
    # Geometric mean is the standard for RF/Filter design
    f0_geom = np.sqrt(fL * fH)
    
    # Arithmetic mean is the "simple" center
    f0_arith = (fL + fH) / 2
    
    return {
        "fl": fL,
        "fH": fH,
        "f0": f0_geom,      # Standard center frequency
        "f0_mid": f0_arith, # Midpoint frequency
        "bw": fH - fL       # Bandwidth
    }