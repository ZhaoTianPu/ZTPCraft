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
            # split on commas first
            items = line.strip().split(", ")
            # Remove a leading "{" from the first token (if present)
            items[0] = items[0].lstrip("{").strip()
            # Remove everything after (and including) the closing "}" in the last token.
            # Using split instead of fixed slicing makes it tolerant to trailing whitespace
            items[-1] = items[-1].split("}")[0].strip()
            array.append([float(item) for item in items])
        return np.array(array)
