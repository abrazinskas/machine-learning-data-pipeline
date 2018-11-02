import numpy as np


class BaseChunkCollector(object):
    """
    Absorbs data-chunks until it gets full, then yields a size adjusted
    data-chunk. 
    
    Can be used for different kinds of groupings or data-units count adjustments.
    """

    def __init__(self, max_size):
        self.max_size = max_size

    @property
    def chunk(self):
        """Returns a compiled data-chunk."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def full(self):
        """Detects if the collector is full."""
        return self.max_size == len(self)

    def absorb_and_yield_if_full(self, data_chunk):
        """
        Adds the data-chunk to the collector, yields a new data_chunk if the
        collector is full.
        """
        raise NotImplementedError

    def _validate_input_length(self, value):
        if len(self) and len(value) != len(self):
            raise ValueError(
                "The size of the input value must be %d,"
                " got %d instead." % (len(self), len(value)))

    @staticmethod
    def _validate_input_value(value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "The input value must be np.ndarray, got %s instead." %
                type(value).name)
