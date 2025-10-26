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

