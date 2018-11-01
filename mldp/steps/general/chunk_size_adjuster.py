from mldp.steps import BaseStep
from chunk_collectors import StandardChunkCollector


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

    def __init__(self, new_size=2, collector_class=StandardChunkCollector,
                 collector_kwargs=None, **kwargs):
        # TODO: docstring, please
        super(ChunkSizeAdjuster, self).__init__(**kwargs)
        self.new_size = new_size

        collector_kwargs = {} if collector_kwargs is None else collector_kwargs
        self._collector = collector_class(max_size=new_size, **collector_kwargs)

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
