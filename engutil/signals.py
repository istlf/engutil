import numpy as np

def generate_sine(frequency, phase, length, amplitude=1, fs=44100):
    t = np.arange(0, length, 1/fs)    
    x = amplitude * np.sin(2*np.pi * frequency * t + phase)
    return t, x

def generate_square(T_0, length, amplitude=None, fs=44100):
    time_vector = np.arange(0, length, 1/fs)
    samples = np.zeros(int(fs * length))
    if amplitude is None:
        amplitude = 1 / T_0
    samples[:int(T_0 * fs)] = amplitude
    return time_vector, samples

def generate_ramp(T_0, length, fs=44100):
    time_vector = np.arange(0, length, 1/fs)
    samples = np.zeros(int(fs * length))
    samples[:int(T_0 * fs)] = time_vector[:int(T_0 * fs)]
    return time_vector, samples

def generate_time(length, fs=44100):
    """
        Generate a time vector.

        Parameters
        ----------
        length : int
            Number of samples.
        fs : int, optional
            Sampling frequency in Hz. Default is 44100.

        Returns
        -------
        numpy.ndarray
            Array of time values in seconds.

        Examples
        --------
        >>> t = generate_time(5, fs=2)
        >>> t
        array([0. , 0.5, 1. , 1.5, 2. ])
    """
    return np.arange(0, length/fs, 1/fs)

def make_spectrum(x, fs, scaling=False, oneside=False):
    """
       freq, Y, YDB = engutil.make_spectrum(x, fs, scaling=False, oneside=False)
        Calculates the frequency spectrum of a signal with correct scaling.

        If 'oneside' and 'scaling' are both True, it computes a one-sided 
        amplitude spectrum. Otherwise, it computes a standard FFT.

        Args:
            x (array-like): Input signal array.
            fs (int or float): Sampling frequency.
            scaling (bool): If True, applies amplitude scaling.
            oneside (bool): If True, returns a one-sided spectrum.

        Returns:
            tuple: A tuple containing (freq, Y, YDB)
                - freq (np.ndarray): Frequency vector.
                - Y (np.ndarray): Complex FFT result (scaled if requested).
                - YDB (np.ndarray): FFT result in decibels.

       
    """
    x = np.asarray(x)
    N = len(x)

    if oneside:
        # Use rfft for efficiency with real signals, as it computes
        # only the positive frequency components.
        Y = np.fft.rfft(x)
        freq = np.fft.rfftfreq(N, d=1 / fs)

        if scaling:
         
            Y = Y / N
            
            Y[1:] *= 2

            if N % 2 == 0:
               
                Y[-1] /= 2
    else:
        # For a standard two-sided spectrum
        Y = np.fft.fft(x)
        freq = np.fft.fftfreq(N, d=1 / fs)
        if scaling:
            # For a two-sided spectrum, the standard amplitude scaling is 1/N.
            Y = Y / N

    # Calculate decibels for the final spectrum.
    # A small constant is added to avoid an error from log10(0).
    YDB = 20 * np.log10(np.abs(Y) + 1e-9)

    return freq, Y, YDB