import unittest
from common import generate_data_chunk
from itertools import product
import numpy as np


class TestDataChunks(unittest.TestCase):
    """Tests different aspects of the data-chunk class."""

    def test_valid_chunks(self):
        chunk_sizes = [1, 20, 12, 1023, 100]
        attrs_numbers = [1, 3, 10, 25, 9]

        for attrs_number, chunk_size in product(attrs_numbers, chunk_sizes):
            good_chunk = generate_data_chunk(attrs_number, chunk_size)
            try:
                good_chunk.validate()
            except StandardError:
                raise self.assertTrue(False)

    def test_chunks_with_wrong_value_types_in_constr(self):
        """Testing if an error is thrown for invalid chunk value types"""
        chunk_size = 100
        attrs_numbers = [2, 3, 10]
        invalid_values = ["dummy_val", [1231, 123123, 12], (), object, 1.23]

        for attrs_number, invalid_val in product(attrs_numbers, invalid_values):
            chunk = generate_data_chunk(attrs_number, chunk_size)
            attr_to_alter = np.random.choice(chunk.keys(), 1)[0]
            with self.assertRaises(TypeError):
                chunk[attr_to_alter] = invalid_val

    def test_chunks_with_different_value_array_sizes(self):
        chunk_size = 100
        attrs_numbers = [2, 3, 10, 25, 9]

        for attrs_number in attrs_numbers:
            chunk = generate_data_chunk(attrs_number, chunk_size)
            attr_to_alter = np.random.choice(chunk.keys(), 1)[0]
            chunk[attr_to_alter] = chunk[attr_to_alter][:-1]
            with self.assertRaises(ValueError):
                chunk.validate()


if __name__ == '__main__':
    unittest.main()
