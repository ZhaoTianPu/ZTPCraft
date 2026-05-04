import numpy as np


def tunneling_strength_from_single_junction(
    EJ: float, EC: float, offset_charge_sum: float
) -> complex:
    """
    Calculate the tunneling strength from a single junction.

    Parameters
    ----------
    EJ: float
        The Josephson energy of the junction.
    EC: float
        The capacitance of the junction.
    offset_charge_sum: float
        The sum of the offset charges of the junction.

    Returns
    -------
    complex
        The tunneling strength.
    """
    return (
        2
        * np.sqrt(2 / np.pi)
        * np.sqrt(8 * EC * EJ)
        * (8 * EJ / EC) ** (1 / 4)
        * np.exp(-np.sqrt(8 * EJ / EC))
        * np.exp(2 * np.pi * 1j * offset_charge_sum)
    )
