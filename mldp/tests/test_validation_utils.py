import unittest
from mldp.utils.util_funcs.validation import validate_data_chunk
import numpy as np
from common import generate_data_chunk
from itertools import product


class TestValidationUtils(unittest.TestCase):

    def test_valid_chunks(self):
        chunk_sizes = [1, 20, 12, 1023, 100]
        attrs_numbers = [1, 3, 10, 25, 9]

        for attrs_number, chunk_size in product(attrs_numbers, chunk_sizes):
            good_chunk = generate_data_chunk(attrs_number, chunk_size)
            try:
                validate_data_chunk(good_chunk)
            except StandardError:
                raise self.assertTrue(False)

    def test_chunks_with_wrong_value_type(self):
        chunk_size = 100
        attrs_numbers = [2, 3, 10]
        invalid_values = ["dummy_val", [1231, 123123, 12], (), object, 1.23]

        for attrs_number, invalid_val in product(attrs_numbers, invalid_values):
            chunk = generate_data_chunk(attrs_number, chunk_size)
            attr_to_alter = np.random.choice(chunk.keys(), 1)[0]
            chunk[attr_to_alter] = invalid_val
            with self.assertRaises(TypeError):
                validate_data_chunk(chunk)

    def test_chunks_with_different_value_array_sizes(self):
        chunk_size = 100
        attrs_numbers = [2, 3, 10, 25, 9]

        for attrs_number in attrs_numbers:
            chunk = generate_data_chunk(attrs_number, chunk_size)
            attr_to_alter = np.random.choice(chunk.keys(), 1)[0]
            chunk[attr_to_alter] = chunk[attr_to_alter][:-1]
            with self.assertRaises(ValueError):
                validate_data_chunk(chunk)


if __name__ == '__main__':
    unittest.main()
