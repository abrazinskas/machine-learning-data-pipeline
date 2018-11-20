from collections import OrderedDict
import numpy as np
from mldp.utils.util_funcs.validation import equal_vals


class DataChunk(object):
    """
    A collection of data units that are passed along the data pipeline.
    Essentially, it's a dictionary of numpy arrays where the first dimension is
    the same along all arrays (i.e. same length).
    """

    def __init__(self, dct=None, preserve_order=True):
        """
        :param dct: dict or ordered dict with numpy arrays of same length
                    (1st dim) that will be used to populate the chunk.
        :param preserve_order: whether to remember the order of inserted fields
                               by using ordered dicts.
        """
        self.data = OrderedDict() if preserve_order else {}
        if dct:
            allowed_types = (dict, OrderedDict)
            if not isinstance(dct, allowed_types):
                rpr_list = repr_types(allowed_types)
                raise TypeError("dct must be a %s" % (" or ".join(rpr_list)))

            for k in dct:
                self[k] = dct[k]
            if not self.is_valid():
                raise ValueError("dct must contain numpy arrays of the same"
                                 " length (1st dim).")

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def append(self, data_unit):
        allowed_types = (dict, OrderedDict)
        rpr = repr_types(allowed_types)
        if isinstance(data_unit, allowed_types):
            raise TypeError("data_unit must be %s" % " or ".join(rpr))
        if not self.is_valid():
            raise ValueError("Can't insert a new data-unit to an invalid"
                             " data-chunk.")
        for k in data_unit:
            if k not in self:
                self[k] = np.array([data_unit[k]])
            else:
                self[k] = np.append(self[k], data_unit[k])

    def __eq__(self, other):
        if not isinstance(other, DataChunk):
            return False

        if len(self.keys()) != len(other.keys()):
            return False

        for k in self.keys():
            if k not in other:
                return False
            if not equal_vals(self[k], other[k]):
                return False

        return True

    @property
    def size(self):
        return len(self)

    @property
    def field_names(self):
        return self.keys()

    def __len__(self):
        if len(self.keys()):
            return len(self[self.keys()[0]])
        return 0

    def __str__(self):
        return str(self.data.items())

    def iter(self):
        """Creates a generator of data-units (dict or ordered dict)."""
        if not self.is_valid():
            raise ValueError("Can't iterate over an invalid data-chunk.")
        for i in range(len(self)):
            data_unit = OrderedDict() if isinstance(self.data, OrderedDict) \
                else {}
            for k in self.keys():
                data_unit[k] = self[k][i]
            yield data_unit

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __delitem__(self, key):
        del self.data[key]

    def __setitem__(self, key, value):
        """Setting the key with arbitrary type of value."""
        self.data[key] = value

    def write(self, f, sep='\t', form_funcs=None):
        """
        Writes the chunk to a file in the tabular format.
        :param f: opened file where the data-chunk has to be written
        :param sep: self-explanatory.
        :param form_funcs: dict of field names mapping to functions that
                                 should be used to obtain str. reprs of values.
        """
        self.validate()
        form_funcs = form_funcs if form_funcs else {}
        if f.tell() == 0:
            f.write(sep.join(self.keys()) + "\n")
        for indx in range(len(self)):
            strs = []
            for fn, fv in self.items():
                form_func = str if fn not in form_funcs else form_funcs[fn]
                cfv = fv[indx]
                strs.append(form_func(cfv))
            f.write(sep.join(strs) + "\n")

    def is_valid(self):
        try:
            self.validate()
        except StandardError:
            return False
        return True

    def validate(self):
        """Checks if all field values are arrays of the same length. """
        not_np_error_mess = "Data-chunk field values must be numpy arrays," \
                            " while '%s' field contains: '%s'."
        not_same_len_error_mess = "All data-chunk field value arrays must be" \
                                  " of the same size."
        prev_len = None
        for k, v in self.items():
            curr_len = len(v)
            if not isinstance(v, np.ndarray):
                raise TypeError(not_np_error_mess % (k, type(v).__name__))
            if prev_len is not None and prev_len != curr_len:
                raise ValueError(not_same_len_error_mess)
            prev_len = curr_len


def repr_types(types):
    return [at.__name__ for at in types]
