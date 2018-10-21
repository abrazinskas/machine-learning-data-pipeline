import unittest
from mldp.steps.readers.csv_reader import CsvReader
from mldp.pipeline import Pipeline
from mldp.steps.transformers import FieldsSelector, FunctionApplier
from mldp.steps.general import ChunkSizeAdjuster
from mldp.steps.formatters.pandas_formatter import PandasFormatter
import numpy as np


class TestDataPipelineIntegration(unittest.TestCase):
    """
    Tests different integration aspects related to the data pipeline.
    Namely, how different combinations of steps operate together being wrapped
    by the data pipeline.
    """

    def test_simple_scenario(self):
        """
        Tries to run the pipeline, and if it works - it's considered to be
        successful. Tries different numbers of workers.
        """
        data_path = 'data/small_chunks'
        field_names = ['first_name', 'email']
        worker_processes_nums = [0, 1, 2, 3, 4]

        reader = CsvReader()

        for wpn in worker_processes_nums:

            dev_data_pipeline = Pipeline(reader,
                                         worker_processes_num=wpn)
            dev_data_pipeline.add_step(FieldsSelector(field_names))
            dev_data_pipeline.add_step(ChunkSizeAdjuster(new_size=10))
            dev_data_pipeline.add_step(PandasFormatter())

            for data_chunk in dev_data_pipeline.iter(data_path=data_path):
                pass

        self.assertTrue(True)

    def test_invalid_pipeline(self):
        """
        Tries to create an invalid data processing pipeline, and expect to get
        an error.
        """
        reader = CsvReader()
        with self.assertRaises(ValueError):
            data_pipeline = Pipeline(reader)
            data_pipeline.add_step(FieldsSelector(["dummy"]))
            data_pipeline.add_step(PandasFormatter())
            data_pipeline.add_step(FunctionApplier({"dummy": lambda x: x}))


if __name__ == '__main__':
    unittest.main()
