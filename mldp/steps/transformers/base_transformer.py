from mldp.utils.util_funcs.validation import validate_data_chunk
from mldp.utils.util_funcs.data_chunks import get_chunk_len
from mldp.steps.base_step import BaseStep


class BaseTransformer(BaseStep):
    """
    Transformers are altering values of data-chunks (e.g. log transformation)
    or filtering their attributes.

    The transformers in most cases apply transformations by reference, namely
    the initial data_chunk's value that was passed as input is not copied, but
    rather modified, and returned.

    It should not change the format of data-chunks, formatters should be used
    for that.
    """

    def __call__(self, data_chunk):
        """
        :type data_chunk: dict of np.ndarray(s)
        :rtype data_chunk: dict of np.ndarray(s)
        """
        try:
            validate_data_chunk(data_chunk, error_mess_prefix='input')
        except StandardError as e:
            raise e
        if get_chunk_len(data_chunk) == 0:
            return data_chunk
        data_chunk = self._transform(data_chunk)
        try:
            validate_data_chunk(data_chunk, error_mess_prefix='output')
        except StandardError as e:
            raise e
        return data_chunk

    def _transform(self, data_chunk):
        """
        :type data_chunk: dict of np.ndarray(s)
        :return: transformed data-chunk (same format) with modified values
        """
        raise NotImplementedError
