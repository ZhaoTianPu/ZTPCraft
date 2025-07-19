import numpy as np
from numpy.typing import NDArray
from typing import List


def read_mathematica_array_file(file_path: str) -> NDArray[np.float64]:
    # the file takes form of this:
    # {3.2, 0.000541363783364755, ...}
    # {3.205, 0.0007340667529176744, ...}
    # {3.21, 0.0010018123277788316, ...}
    # ...
    with open(file_path, "r") as f:
        lines = f.readlines()
        array: List[List[float]] = []
        for line in lines:
            items = line.split(", ")
            items[0] = items[0][1:]  # remove the first {
            items[-1] = items[-1][:-2]  # remove the last }
            array.append([float(item) for item in items])
        return np.array(array)
