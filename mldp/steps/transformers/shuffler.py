from mldp.steps.transformers.base_transformer import BaseTransformer
from mldp.utils.util_funcs.data_chunks import get_chunk_len
from numpy.random import permutation, seed as seed_func


class Shuffler(BaseTransformer):
    """
    Shuffles data chunk values. Can be used to break sequential order dependency
    of data-units.

    A straightforward application of the shuffler is to shuffle large data-chunk
    produced by a reader and afterwards batch data units into smaller data-chunks

    For example, the architecture could look like this:
        Reader -> Shuffler -> ValueTransformer -> ChunkSizeAdjuster -> Formatter
    """

    def __init__(self, seed=None, **kwargs):
        super(Shuffler, self).__init__(**kwargs)
        if seed is not None:
            seed_func(seed)

    def _transform(self, data_chunk):
        """
        :param data_chunk: dict of np.ndarrays
        :return: data-chunk with shuffled values
        """
        chunk_len = get_chunk_len(data_chunk)
        shuffled_order = permutation(range(chunk_len))
        for key in data_chunk:
            data_chunk[key] = data_chunk[key][shuffled_order]
        return data_chunk
