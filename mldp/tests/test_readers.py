import unittest
from mldp.steps.readers import CsvReader
from mldp.tests.common import read_from_csv_file
import numpy as np


class TestReaders(unittest.TestCase):

    def test_csv_reader_valid_paths(self):
        """
        Passing an intentionally wrong input to the reader and expecting it to
        throw an error.
        """
        data_paths = ["a", "b", 123123123, 123.12313]
        reader = CsvReader()
        itr = reader.iter(data_path=data_paths)

        with self.assertRaises(ValueError):
            chunk = next(itr.__iter__())

    def test_csv_reader_output(self):
        """ Checking if read data-chunks are actually valid. """
        data_path = 'data/small_chunks/chunk2.csv'
        chunk_size = 2

        reader = CsvReader(chunk_size=chunk_size)

        expected_data_chunks = read_from_csv_file(data_path)

        for chunk_indx, chunk in enumerate(reader.iter(data_path=data_path)):
            for i in range(chunk_size):
                el_indx = chunk_indx * chunk_size + i
                for key in chunk.keys():
                    expt_val = expected_data_chunks[key][el_indx]
                    self.assertTrue(chunk[key][i] == expt_val)

    def test_csv_multi_threaded_reader_output(self):
        """
        Check if multi-threaded and single threaded readers produce the correct
        output.
        """
        data_paths = ['data/small_chunks/chunk1.csv',
                      'data/small_chunks/chunk2.csv',
                      'data/small_chunks/chunk3.csv',
                     ]
        chunk_size = 2

        reader = CsvReader(chunk_size=chunk_size)

        expected_data_chunks = read_from_csv_file(data_paths)

        actual_data_chunks = {}
        for data_chunk in reader.iter(data_path=data_paths):
            for key in data_chunk.keys():
                if key not in actual_data_chunks:
                    actual_data_chunks[key] = np.array([])
                actual_data_chunks[key] = np.concatenate([
                                                        actual_data_chunks[key],
                                                        data_chunk[key]
                                                        ])
        self.compare_unsorted_data_chunks(dc1=expected_data_chunks,
                                          dc2=actual_data_chunks,
                                          sort_key='id')

    def compare_unsorted_data_chunks(self, dc1, dc2, sort_key):
        # check keys
        self.assertTrue(len(dc1) == len(dc2))
        for key in dc1.keys():
            self.assertTrue(key in dc2)

        sort_order1 = np.argsort(dc1[sort_key])
        sort_order2 = np.argsort(dc2[sort_key])

        for key in dc1.keys():
            self.assertTrue(np.array_equal(dc1[key][sort_order1],
                                           dc2[key][sort_order2]))


if __name__ == '__main__':
    unittest.main()
