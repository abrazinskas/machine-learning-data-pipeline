from mldp.steps.formatters.base_formatter import BaseFormatter
import pandas


class PandasFormatter(BaseFormatter):
    """
    Converts data-chunks to pandas data-frames.
    """

    def _format(self, data_chunk):
        """
        :param data_chunk: dict of np.ndarray(s)
        :return: Pandas DataFrame.
        """
        return pandas.DataFrame(data_chunk)
