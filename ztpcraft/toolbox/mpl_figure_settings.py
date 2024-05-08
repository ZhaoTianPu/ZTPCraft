import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Union
from ztpcraft.util.figure_settings import MPL_PRESET


def update_matplotlib_settings(settings: Union[Dict[str, Any], str]):
    """
    Update the matplotlib settings.

    Parameters
    ----------
    settings : Union[Dict[str, Any], str]
        The settings to update the matplotlib. If a string is given, it should
        be a key in the MPL_PRESET dictionary.
    """
    if isinstance(settings, str):
        try:
            settings = MPL_PRESET[settings]
        except KeyError:
            raise KeyError(f"settings {settings} not found in MPL_PRESET.")
    plt.rcParams.update(settings)
