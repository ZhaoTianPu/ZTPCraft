import time
import os

import dill
import h5py
import numpy as np
import pandas as pd

from typing import Any, Dict, Literal


def datetime_dir(
    save_dir="./",
    dir_suffix=None,
):
    """
    Initialize a directory with the current datetime.

    Parameters & Examples
    ---------------------
    save_dir : str
        The directory to save the data, default to be "./". Say the current
        datetime is 2021-01-31 12:34, then the directory will be
        "save_dir/20210131/12_34/".
    dir_suffix : str
        The suffix of the directory, default to be None. Say the current
        datetime is 2021-01-31 12:34, then the directory will be
        "save_dir/20210131/12_34_dir_suffix/".

    Returns
    -------
    current_date_dir : str
    """
    save_dir = os.path.normpath(save_dir)

    current_time = time.localtime()
    current_ymd_dir = save_dir + time.strftime("/%Y%m%d/", current_time)
    current_time_dir = current_ymd_dir + time.strftime("%H_%M", current_time)

    if dir_suffix != "" and dir_suffix is not None:
        current_date_dir = current_time_dir + "_" + dir_suffix + "/"
    else:
        current_date_dir = current_time_dir + "/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(current_ymd_dir):
        os.mkdir(current_ymd_dir)
    if not os.path.exists(current_date_dir):
        os.mkdir(current_date_dir)

    print(f"Current datetime directory: {current_date_dir}")
    return current_date_dir


def dill_dump(obj: Any, filename: str) -> None:
    """Dump a python object to a file using dill."""
    filename = os.path.normpath(filename)
    file = open(filename, "wb")
    dill.dump(obj, file)
    file.close()


def dill_load(filename: str) -> Any:
    """Load a python object from a file using dill."""
    filename = os.path.normpath(filename)
    file = open(filename, "rb")
    obj = dill.load(file)
    file.close()

    return obj
