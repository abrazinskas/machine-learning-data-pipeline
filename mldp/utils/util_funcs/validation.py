from mldp.utils.util_funcs.general import all_elements_are_equal
from general import listify
import numpy as np
from collections import OrderedDict


def validate_data_chunk(data_chunk, error_mess_prefix=""):
    """Checks if data-chunk is a dictionary of the same size numpy arrays."""
    if error_mess_prefix != "":
        error_mess_prefix = error_mess_prefix[0].upper() + error_mess_prefix[1:]

    if not isinstance(data_chunk, dict):
        error_mess = "data-chunk must be '%s', while it's '%s'." % \
                     (dict.__name__, type(data_chunk).__name__)
        if error_mess_prefix != "":
            error_mess = error_mess_prefix + " " + error_mess
        else:
            error_mess = error_mess[0].upper() + error_mess[1:]
        raise TypeError(error_mess)
    lens = []
    for k, v in data_chunk.items():
        curr_len = len(v)
        if not isinstance(v, np.ndarray):
            error_mess = "data-chunk field values must be numpy arrays, while" \
                         " '%s' field contains: '%s'." %\
                         (k, type(v).__name__)
            if error_mess_prefix != "":
                error_mess = error_mess_prefix + " " + error_mess
            else:
                error_mess = error_mess[0].upper() + error_mess[1:]
            raise TypeError(error_mess)
        lens.append(curr_len)

    if not all_elements_are_equal(lens):
        raise ValueError("All data-chunk arrays must be of the same size.")


def validate_field_names(field_names):
    """Checks whether field_names is either a string or a list of strings."""
    field_name_error_mess = "Please provide valid field_names." \
                            " It must be either a string or list of strings."
    if isinstance(field_names, list):
        for field_name in field_names:
            if not isinstance(field_name, (unicode, str)):
                raise ValueError(field_name_error_mess)
        return
    if not isinstance(field_names, (unicode, str)):
        raise ValueError(field_name_error_mess)


def validate_field_names_mapping(field_names_to_something, value_types):
    error_mess = "Please provide a valid mapping (dict, OrderedDict) from" \
                 " strings/unicode to "
    if isinstance(value_types, tuple):
        error_mess += ", ".join([v.__name__ for v in value_types])
    else:
        error_mess += value_types.__name__
    error_mess += " objects"
    if not isinstance(field_names_to_something, (dict, OrderedDict)):
        raise ValueError(error_mess)
    for key, val in field_names_to_something.items():
        if not isinstance(key, (str, unicode)) \
                or not isinstance(val, value_types):
            raise ValueError(error_mess)


def validate_data_paths(data_paths):
    """ Validates path(s) by checking their type. """
    if not isinstance(data_paths, (str, unicode, list)):
        raise ValueError("Please provide a valid data_path(s)")
    if isinstance(data_paths, list):
        for i, p in enumerate(data_paths):
            if not isinstance(p, (str, unicode)):
                raise ValueError("Data path at the position %d is invalid"
                                 % i)


def equal_to_constant(var, constant):
    if type(var) != type(constant):
        return False
    return var == constant
