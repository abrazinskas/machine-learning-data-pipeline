from mldp.steps import BaseStep
from mldp.utils.util_classes import DataChunk
from collections import OrderedDict
import numpy as np


class ChunkSizeAdjuster(BaseStep):
    """
    ChunkSizeAdjuster step allows to change the size of data-chunks that are
    passed along the pipeline.
    The step does not alter the format of data-chunks, only their size.

    For example, one might want to use larger chunks (e.g. size of 500) for
    computational purposes (fast vectorized operations on large numpy arrays)
    but to train a model on smaller data-chunks (e.g. size of 64). In that case,
    the step should be added after all computationally intensive ones.

    It works both by accumulating smaller upstream data-chunks and passing
    larger data-chunks downstream, and splitting larger upstream data-chunks
    into smaller downstream data-chunks.
    """

    def __init__(self, new_size=2, **kwargs):
        super(ChunkSizeAdjuster, self).__init__(**kwargs)
        self.new_size = new_size
        self._collector = _ChunkCollector(max_size=new_size)

    def iter(self, data_chunk_iter):
        """
        Wraps the data-chunk iterable into a generator that yields data-chunks
        with the adjusted size.
        """
        for data_chunk in data_chunk_iter:
            try:
                data_chunk.validate()
            except Exception as e:
                raise e

            for adjusted_dc in self._collector.absorb_yield_if_full(data_chunk):
                yield adjusted_dc

        # yield the last incomplete chunk
        if len(self._collector):
            yield self._collector.chunk


class _ChunkCollector(object):
    """
    Absorbs data-chunks until it gets full, then yields a size adjusted
    data-chunk. Automatically resets itself after yielding.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        # key: field_names, values: chunk field values in a list
        self._chunk_data_collector = OrderedDict()

    @property
    def chunk(self):
        """Returns a data-chunk."""
        dc = DataChunk()
        for k, v in self._chunk_data_collector.items():
            dc[k] = np.concatenate(v)
        return dc

    def full(self):
        """Detects if the collector is full."""
        return self.max_size == len(self)

    def append(self, k, v):
        self._validate_input_value(v)
        if k not in self._chunk_data_collector:
            self._chunk_data_collector[k] = []
        self._chunk_data_collector[k].append(v)

    def __getitem__(self, key):
        return self._chunk_data_collector[key]

    # def __setitem__(self, key, value):
    #     self._validate_input_value(value)
    #     self._validate_input_length(value)
    #     self._chunk_data_collector[key] = value

    def __len__(self):
        keys = self._chunk_data_collector.keys()
        if len(keys) == 0:
            return 0
        return sum([len(el) for el in self._chunk_data_collector[keys[0]]])

    def absorb_yield_if_full(self, data_chunk):
        """
        Adds the data-chunk to the collector, yields a new data_chunk if the
        collector is full.
        """
        start_indx = 0
        end_indx = len(data_chunk)

        while start_indx < end_indx:
            size_before = len(self)
            missing_count = self.max_size - size_before
            tmp_end_indx = min(start_indx + missing_count, end_indx)
            self.collect_missing_units(data_chunk, start_indx,
                                       end_indx=tmp_end_indx)
            start_indx += (len(self) - size_before)

            # if it's full yield and reset
            if self.full():
                yield self.chunk
                self._chunk_data_collector = {}

    def collect_missing_units(self, data_chunk, start_indx, end_indx):
        """Stores units from the data-chunk to the collector."""
        slice_indx = range(start_indx, end_indx)
        for k in data_chunk:
            self.append(k, data_chunk[k][slice_indx])

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
