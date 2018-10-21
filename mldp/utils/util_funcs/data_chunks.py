from numpy import isclose
import numpy as np


def equal_data_chunks(data_chunk1, data_chunk2):
    """
    Checks if both keys and values (arrays) of dictionaries/chunks match.
    Works for multidimensional field values.
    """
    if len(data_chunk1) != len(data_chunk2):
        return False

    if len(data_chunk1.keys()) != len(data_chunk2.keys()):
        return False

    for k in data_chunk1:
        if k not in data_chunk2:
            return False
        if not equal_vals(data_chunk1[k], data_chunk2[k]):
            return False

    return True


def equal_vals(val1, val2):
    """Recursively checks equality of two values."""
    if type(val1) != type(val2):
        return False
    if isinstance(val1, (list, tuple, np.ndarray)):
        if len(val1) != len(val2):
            return False
        for indx in range(len(val1)):
            if not equal_vals(val1[indx], val2[indx]):
                return False
        return True
    elif isinstance(val1, float):
        return isclose(val1, val2)
    else:
        return val1 == val2


def get_chunk_len(data_chunk):
    """Returns the length of the data_chunk assuming it's a valid one """
    return len(data_chunk[data_chunk.keys()[0]])
