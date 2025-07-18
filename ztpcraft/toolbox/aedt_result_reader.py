import re
import numpy as np
from typing import Dict, Literal, Union, Tuple
from numpy.typing import NDArray


def read_hfss_eigenmode_file(
    filename: str, return_loss_as: Literal["Q", "imag_freq"] = "Q"
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Read the eigenmode file from HFSS.

    Parameters
    ----------
    filename : str
        The path to the eigenmode file.
    return_loss_as : Literal["Q", "imag_freq"], optional
        The loss to be returned. If "Q", the Q factor is returned. If "imag_freq", the imaginary part of the frequency is returned.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        If return_loss_as is "Q", a tuple of two arrays, the first is the frequencies, the second is the Q factors.
        If return_loss_as is "imag_freq", a tuple of two arrays, the first is the frequencies, the second is the imaginary part of the frequency.
        If return_loss_as is not "Q" or "imag_freq", an error is raised.
    """
    with open(filename, "r") as file:
        lines = file.readlines()[7:]  # Skip the first 7 lines

    # Regular expression pattern for matching the data lines
    pattern = r"\s*(\d+)\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+(?:\+\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+j\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)|([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?))"

    frequencies: list[float] = []
    Q_factors: list[float] = []
    imag_freqs: list[float] = []
    is_lossy = False
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # lossy case
            # copy the match.groups(), remove all the None values, and check the length
            match_result = [i for i in match.groups() if i is not None]
            if len(match_result) == 4:
                # Extract the data from the matched groups
                (
                    _,
                    real_freq,
                    imag_freq,
                    Q,
                ) = match_result
                frequencies.append(float(real_freq))
                imag_freqs.append(float(imag_freq))
                Q_factors.append(float(Q))
                is_lossy = True
            # lossless case
            elif len(match_result) == 2:
                # Extract the data from the matched groups
                _, real_freq = match_result
                frequencies.append(float(real_freq))
                is_lossy = False
            else:
                raise ValueError(f"Unexpected number of matches: {len(match_result)}")
    if return_loss_as == "Q":
        if is_lossy:
            return np.array(frequencies), np.array(Q_factors)
        else:
            return np.array(frequencies)
    elif return_loss_as == "imag_freq":
        if is_lossy:
            return np.array(frequencies), np.array(imag_freqs)
        else:
            return np.array(frequencies)
    else:
        raise ValueError(f"Invalid return_loss_as: {return_loss_as}")


def weighted_sum(
    indexed_results: Dict[int, float], idx_weight_dict: Dict[int, float]
) -> float:
    """
    A function to calculate the weighted sum of a dictionary of indexed results.

    Parameters
    ----------
    indexed_results : Dict[int, float]
        A dictionary of indexed results.
    idx_weight_dict : Dict[int, float]
        A dictionary of index and corresponding weight.

    Returns
    -------
    float
        The weighted sum of the indexed results.
    """
    weighted_total = 0
    for idx, weight in idx_weight_dict.items():
        weighted_total += indexed_results[idx] * weight
    return weighted_total
