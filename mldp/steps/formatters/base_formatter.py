from mldp.steps.base_step import BaseStep


class BaseFormatter(BaseStep):
    """
    Responsible for format transformations of data-chunks, e.g. conversion to
    pandas data frames. For value adjustments, please use transformers.
    """
    def __init__(self, **kwargs):
        super(BaseFormatter, self).__init__(**kwargs)

    def __call__(self, data_chunk):
        """
        :param data_chunk: dict of np.ndarrays.
        :return: depends on the children class's _format(), without any
                 format restrictions.
        """
        try:
            data_chunk.validate()
        except StandardError as e:
            raise e

        return self._format(data_chunk)

    def _format(self, data_chunk):
        """
        :param data_chunk: dict of np.ndarrays.
        :return: data_chunk in a desired format (e.g. pandas data-frame).
        """
        raise NotImplementedError
