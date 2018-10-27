import unittest
from mldp.utils.util_funcs.validation import validate_data_chunk
import numpy as np
from common import generate_data_chunk
from itertools import product


class TestValidationUtils(unittest.TestCase):



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
