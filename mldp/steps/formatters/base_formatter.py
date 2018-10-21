from mldp.utils.util_funcs.validation import validate_data_chunk
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
        :param data_chunk: dict of np.ndarray(s).
        :return: depends on the children class's _format(), without any
                 format restrictions.
        """
        try:
            validate_data_chunk(data_chunk, error_mess_prefix="input")
        except StandardError as e:
            raise e
        return self._format(data_chunk)

    def _format(self, data_chunk):
        """
        :param data_chunk: dict of np.ndarray(s).
        :return: data_chunk in a desired format(e.g. pandas data-frame).
        """
        raise NotImplementedError
