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
        try:
            data_chunk.validate()
        except StandardError as e:
            raise e
        if len(data_chunk) == 0:
            return data_chunk

        data_chunk = self._transform(data_chunk)
        try:
            data_chunk.validate()
        except StandardError as e:
            raise e
        return data_chunk

    def _transform(self, data_chunk):
        """
        :param data_chunk: self-explanatory.
        :return: transformed data-chunk with modified field values.
        """
        raise NotImplementedError
