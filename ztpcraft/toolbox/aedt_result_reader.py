import re
import numpy as np


def read_hfss_eigenmode_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()[7:]  # Skip the first 7 lines

    # Regular expression pattern for matching the data lines
    pattern = r"\s*(\d+)\s+([-\d.Ee+]+)\s+\+\s+([-\d.Ee+]+)\s+j\s+([-\d.Ee+]+)"

    frequencies = []
    Q_factors = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Extract the data from the matched groups
            index, real_freq, imag_freq, Q = match.groups()
            frequencies.append(float(real_freq))
            Q_factors.append(float(Q))

    return np.array(frequencies), np.array(Q_factors)
