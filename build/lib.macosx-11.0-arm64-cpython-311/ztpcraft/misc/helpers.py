import numpy as np
from typing import List, Any
from numpy.typing import NDArray

def find_closest_index_numpy(lst: List[Any], x: Any) -> int:
    """
    Find the index of the closest element in a list to a given value.

    Parameters:
    - lst: List of floats
    - x: Value to find the closest index to
    """
    arr: NDArray[Any] = np.array(lst)
    index: int = (np.abs(arr - x)).argmin()
    return index