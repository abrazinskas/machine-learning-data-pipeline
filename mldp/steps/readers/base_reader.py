from mldp.utils.util_funcs.validation import validate_data_chunk
from mldp.steps.base_step import BaseStep


class BaseReader(BaseStep):
    """
    Defines the blue-print for children classes that read data from a local and
    remote storage.

    Reader objects can be used as standalone objects for raw data-chunks
    iteration, simply use .iter(**data_source_params). Alternatively, it can be
    used in a pipeline.
    """

    def __init__(self, chunk_size=1000, **kwargs):
        super(BaseReader, self).__init__(**kwargs)
        self.chunk_size = chunk_size

    def iter(self, **kwargs):
        """
        Creates an iterable generator over data chunks which are created by
        reading data specified by **kwargs.

        :param kwargs: must be coherent with _iter method that is implemented in
                       subclasses. E.g. it might data_path, along with
                       units_per_file that control the units that are read from
                       each file.
        :return: generator over data-chunks.
        """
        for data_chunk in self._iter(**kwargs):
            try:
                validate_data_chunk(data_chunk)
            except Exception as e:
                raise e
            yield data_chunk

    def _iter(self, **kwargs):
        """
        One has to implement the actual logic for data reading in subclass
        readers.

        :return: generator over data-chunks.
        """
        raise NotImplementedError
