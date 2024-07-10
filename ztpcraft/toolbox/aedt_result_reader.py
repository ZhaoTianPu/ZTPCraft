import re
import numpy as np


def read_hfss_eigenmode_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()[7:]  # Skip the first 7 lines

    # Regular expression pattern for matching the data lines
    pattern = r"\s*(\d+)\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+(?:\+\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+j\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)|([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?))"

    frequencies = []
    Q_factors = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            # lossy case
            # copy the match.groups(), remove all the None values, and check the length
            match_result = [i for i in match.groups() if i is not None]
            if len(match_result) == 4:
                # Extract the data from the matched groups
                (
                    index,
                    real_freq,
                    imag_freq,
                    Q,
                ) = match_result
                frequencies.append(float(real_freq))
                Q_factors.append(float(Q))
            # lossless case
            elif len(match_result) == 3:
                # Extract the data from the matched groups
                index, real_freq, Q = match_result
                frequencies.append(float(real_freq))
                Q_factors.append(float(Q))
    return np.array(frequencies), np.array(Q_factors)
