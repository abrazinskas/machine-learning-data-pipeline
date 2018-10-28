from mldp.steps.transformers.base_transformer import BaseTransformer
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
        :param data_chunk: self.explanatory.
        :return: data-chunk with shuffled field values.
        """
        shuffled_order = permutation(range(len(data_chunk)))
        for key in data_chunk.keys():
            data_chunk[key] = data_chunk[key][shuffled_order]
        return data_chunk
