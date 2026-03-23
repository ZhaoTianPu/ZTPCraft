import time
import os

import dill
import h5py

from typing import Any, Dict


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


# Save data to HDF5 file
def h5_dump(
    data_dict: Dict[str, Any],
    file_name: str,
):
    """
    Dump a single-level dictionary to a HDF5 file.

    Note: This function only supports single-level dictionaries. Nested
    dictionaries are not supported and will raise a ValueError.

    Parameters
    ----------
    file_name : str
        The filename of the HDF5 file.
    data_dict : Dict[str, Any]
        A single-level dictionary where values are arrays or scalars,
        not nested dictionaries.

    Raises
    ------
    ValueError
        If any value in data_dict is a dictionary (nested structure).
    """
    # Check for nested dictionaries
    for key, value in data_dict.items():
        if isinstance(value, dict):
            raise ValueError(
                f"Nested dictionaries are not supported. "
                f"Key '{key}' contains a dictionary value. "
                f"Please flatten your data structure."
            )

    with h5py.File(file_name, "w") as f:
        # Create datasets for all data
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)


# Load data from HDF5 file
def h5_load(
    file_name: str,
) -> Dict[str, Any]:
    """
    Load a single-level dictionary from a HDF5 file.

    Note: This function only supports single-level HDF5 files. If the HDF5 file
    contains groups (nested structure), the keys will be flattened using forward
    slashes (e.g., 'group/dataset'). For true nested dictionary support,
    consider using a different approach.

    Parameters
    ----------
    file_name : str
        The filename of the HDF5 file.

    Returns
    -------
    data_dict : Dict[str, Any]
        A single-level dictionary loaded from the HDF5 file. If the original
        file had nested groups, the keys will be flattened (e.g., 'group/dataset').
    """
    data_dict = {}
    with h5py.File(file_name, "r") as f:
        # Helper function to read all items (flattens nested structure)
        def extract_data(name, obj):
            if isinstance(obj, h5py.Dataset):
                data_dict[name] = obj[()]

        # Visit all items in the file
        f.visititems(extract_data)

        # Print file structure
        print("File structure:")

        def print_info(name, obj):
            print(
                f"{name}, shape: {obj.shape}" if isinstance(obj, h5py.Dataset) else name
            )

        f.visititems(print_info)

    return data_dict
