import time
import os

import dill

from typing import Any


def datetime_dir(
    save_dir: str = "./",
    dir_suffix: str | None = None,
    save_time: bool = True,
) -> str:
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
    current_datetime_dir = current_ymd_dir + time.strftime("%H_%M_%S", current_time)

    if save_time:
        if dir_suffix != "" and dir_suffix is not None:
            current_datetime_dir = current_datetime_dir + "_" + dir_suffix + "/"
        else:
            current_datetime_dir = current_datetime_dir + "/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(current_ymd_dir):
        os.mkdir(current_ymd_dir)
    if save_time and not os.path.exists(current_datetime_dir):
        os.mkdir(current_datetime_dir)

    if save_time:
        print(f"Current datetime directory: {current_datetime_dir}")
        return current_datetime_dir
    else:
        print(f"Current date directory: {current_ymd_dir}")
        return current_ymd_dir


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
