from mldp.steps.transformers.base_transformer import BaseTransformer
from mldp.steps.transformers import FunctionApplier, Shuffler, VocabMapper,\
    Padder, WindowSlider, TokenProcessor
from mldp.steps.transformers.common import create_new_field_name
from mldp.steps.readers import CsvReader
from mldp.utils.util_funcs.data_chunks import equal_data_chunks
from mldp.utils.util_classes import Vocabulary
from common import generate_data_chunk, create_list_of_data_chunks,\
    read_from_csv_file
from mock import patch
import copy
from numpy import isclose
from itertools import product
import unittest
import numpy as np
import re


class TestTransformers(unittest.TestCase):

    def test_function_applier(self):
        data_size = 1234
        data_attrs_number = 15
        input_chunks_size = 10
        transform_attrs_number = 10

        functions = [lambda x: np.log(abs(x) + 1), lambda x: np.exp(x),
                     lambda x: x**2]
        data = generate_data_chunk(data_attrs_number, data_size)
        transform_attrs = data.keys()[:transform_attrs_number]
        input_data_chunks = create_list_of_data_chunks(data, input_chunks_size)

        for func in functions:
            function_applier = FunctionApplier({a:func for a in transform_attrs})
            for input_data_chunk in input_data_chunks:
                actual_chunk = function_applier(input_data_chunk)
                expected_chunk = input_data_chunk

                # transforming manually values of input data-chunks
                for transform_attr in transform_attrs:
                    expected_chunk[transform_attr] = \
                        func(expected_chunk[transform_attr])

                self.assertTrue(equal_data_chunks(actual_chunk, expected_chunk))

    def test_input_data_chunk_format_validation(self):
        """
        Testing if the base class of transformers properly validates the input
        data chunks.
        """
        invalid_input_data_chunk = [[1, 2, 3], [5, 6, 8]]

        with patch('mldp.steps.transformers.base_transformer.BaseTransformer')\
                as bf:
            bf._transform = lambda x: x

            with self.assertRaises(StandardError):
                my_bf = BaseTransformer()
                my_bf(invalid_input_data_chunk)

    def test_output_data_chunk_format_validation(self):
        """
        Testing if the base class of transformers properly validates the output
        data chunks.
        """
        valid_input_data_chunk = {"dummy": np.array([1, 2, 3])}

        with patch('mldp.steps.transformers.base_transformer.BaseTransformer')\
                as bf:
            bf._transform = lambda x: []

            with self.assertRaises(StandardError):
                my_bf = BaseTransformer()
                my_bf(valid_input_data_chunk)

    def test_shuffler(self):
        """
        Checks whether shuffler changes the order, and does not eliminate
        elements.
        """
        seed = 10
        data_chunk_sizes = [30, 10, 100, 320]
        data_attrs_numbers = [2, 1, 3, 15]

        for data_chunk_size, data_attrs_number in product(data_chunk_sizes,
                                                          data_attrs_numbers):
            data_chunk = generate_data_chunk(data_attrs_number, data_chunk_size)
            original_data_chunk = copy.deepcopy(data_chunk)
            shuffler = Shuffler(seed=seed)

            # Checking if the order is actually broken for desired fields/attrs,
            # and all data-units are preserved for the shuffled fields
            shuffled_data_chunk = shuffler(data_chunk)

            for attr in shuffled_data_chunk:
                res = isclose(original_data_chunk[attr],
                              shuffled_data_chunk[attr])
                self.assertFalse(res.all())

                res = isclose(sorted(original_data_chunk[attr]),
                              sorted(shuffled_data_chunk[attr]))
                self.assertTrue(res.all())

    def test_vocabulary_mapper(self):
        """Testing whether the mapper allows to map back and forth field values.
        """
        data_path = 'data/mock_data.csv'
        target_fields = ["first_name", "last_name", "email", "gender"]

        reader = CsvReader()
        vocab = Vocabulary(reader)

        for target_field in target_fields:
            vocab.create(data_source={"data_path": data_path},
                         data_field_names=target_field)

            data = read_from_csv_file(data_path)
            data_original = copy.deepcopy(data)

            mapper_to = VocabMapper({target_field: vocab}, "id")
            mapper_back = VocabMapper({target_field: vocab}, "token")

            data = mapper_to(data)
            data = mapper_back(data)

            self.assertTrue((data[target_field] == data_original[target_field])
                            .all())

    def test_vocabulary_mapper_multidim_lists(self):
        """Testing whether the mapper can map multi-dim lists."""
        target_field_name = "dummy"
        symbols_attr = "id"

        data_chunk = {target_field_name: np.array([
            [["one"], ["two"]],
            [["three"], ["four", "five", "six"]]
        ], dtype="object")}
        exp_val = np.empty(2, dtype="object")
        exp_val[0] = np.array([[1], [2]])
        exp_val[1] = np.array([[3], [4, 5, 6]])
        expected_output_chunk = {target_field_name: exp_val}

        # creating and populating a vocab
        vocab = Vocabulary()
        vocab._add_symbol("zero")
        vocab._add_symbol("one")
        vocab._add_symbol("two")
        vocab._add_symbol("three")
        vocab._add_symbol("four")
        vocab._add_symbol("five")
        vocab._add_symbol("six")

        mapper = VocabMapper({target_field_name: vocab},
                             symbols_attr=symbols_attr)
        actual_output_chunk = mapper(copy.deepcopy(data_chunk))

        self.assertTrue(equal_data_chunks(actual_output_chunk,
                                          expected_output_chunk))

    def test_vocabulary_mapper_mixed_field_values(self):
        """Testing whether the mapper can map multi-dim mixed field values."""
        target_field_name = "dummy"
        symbols_attr = "id"

        data_chunk = {target_field_name: np.array([
            [["one"], np.array(["two", "one"])],
            [["three"], np.array(["four", "five", "six"])]
        ], dtype="object")}
        expected_output_chunk = {target_field_name: np.array([
            [[1], np.array([2, 1])],
            [[3], np.array([4, 5, 6])]
        ], dtype="object")}

        # creating and populating a vocab
        vocab = Vocabulary()
        vocab._add_symbol("zero")
        vocab._add_symbol("one")
        vocab._add_symbol("two")
        vocab._add_symbol("three")
        vocab._add_symbol("four")
        vocab._add_symbol("five")
        vocab._add_symbol("six")

        mapper = VocabMapper({target_field_name: vocab},
                             symbols_attr=symbols_attr)
        actual_output_chunk = mapper(data_chunk)

        self.assertTrue(equal_data_chunks(actual_output_chunk,
                                          expected_output_chunk))

    def test_2D_padder(self):
        """
        Testing if padding works correctly for common scenarios of 2D data
        (batch_size x sequences).
        Specifically testing whether it produces proper padded sequences, and
        their masks. Also, testing if when symbol_to_mask is provided if it
        correctly masks those symbols.
        """
        field_names = ["text"]
        data_path = "data/news.txt"
        pad_symbol = "<PAD>"
        mask_field_name_suffix = "mask"
        padding_modes = ['left', 'right', 'both']
        symbols_to_mask = ["The", "a", "to", "as"]
        axis = 1

        data_chunk = read_from_csv_file(data_path, sep="\t")

        # tokenize field values
        for fn in field_names:
            data_chunk[fn] = np.array([seq.split() for seq in data_chunk[fn]])

        for padding_mode, symbol_to_mask in product(padding_modes,
                                                    symbols_to_mask):
            padder = Padder(field_names, pad_symbol=pad_symbol,
                            new_mask_field_name_suffix=mask_field_name_suffix,
                            padding_mode=padding_mode, axis=axis,
                            symbol_to_mask=symbol_to_mask)
            padded_data_chunk = padder(copy.deepcopy(data_chunk))

            for fn in field_names:
                mask_fn = create_new_field_name(fn,
                                                suffix=mask_field_name_suffix)
                padded_fv = padded_data_chunk[fn]
                mask = padded_data_chunk[mask_fn]
                original_fv = data_chunk[fn]

                self.assertTrue(len(padded_fv.shape) == 2)
                self._test_padded_values(original_field_values=original_fv,
                                         padded_field_values=padded_fv,
                                         mask=mask, pad_symbol=pad_symbol,
                                         symbol_to_mask=symbol_to_mask)

    def test_3D_padder(self):
        """Light version test to check if the padder works for 3D data."""
        field_name = "dummy"
        pad_symbol = -99
        mask_fn_suffix = "mask"
        padding_mode = "both"
        axis = 2

        data_chunk = {field_name: np.array([
            [[0, 1, 2], [3, 4, 5], [], [6]],
            [[1], [1, 2], []],
            []
        ])}
        padder = Padder(field_name, pad_symbol=pad_symbol, axis=axis,
                        new_mask_field_name_suffix=mask_fn_suffix,
                        padding_mode=padding_mode)
        padded_data_chunk = padder(copy.deepcopy(data_chunk))

        mask_fn = create_new_field_name(field_name, suffix=mask_fn_suffix)

        original_fv = data_chunk[field_name]
        padded_fv = padded_data_chunk[field_name]
        mask = padded_data_chunk[mask_fn]

        for ofv, pfv, m in zip(original_fv, padded_fv, mask):
            self._test_padded_values(original_field_values=ofv,
                                     padded_field_values=pfv, mask=m,
                                     pad_symbol=pad_symbol)

    def _test_padded_values(self, original_field_values, padded_field_values,
                            mask, pad_symbol, symbol_to_mask=None):
        self.assertTrue(len(padded_field_values) == len(original_field_values))

        # testing both padded sequences and the produced mask
        for seq_act, seq_exp, m in zip(padded_field_values,
                                       original_field_values,
                                       mask):
            indx = -1
            for c_indx, elem in enumerate(seq_act):
                if elem != pad_symbol:
                    indx += 1
                if elem != pad_symbol and elem != symbol_to_mask:
                    self.assertEqual(elem, seq_exp[indx])
                else:
                    self.assertEqual(0., m[c_indx])

    def test_window_slider(self):
        field_name = "dummy"
        suffix = "window"
        new_field_name = create_new_field_name(field_name, suffix=suffix)

        # scenario 1.
        window_size = 2
        step_size = 1
        only_full_windows = False
        input_seqs = np.array([list(range(6)), list(range(2))])
        input_chunk = {field_name: input_seqs}
        expect_seqs = np.array([
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5]],
            [[0, 1]]])
        expected_output = {field_name: input_seqs, new_field_name: expect_seqs}
        self._test_window_setup(input_chunk, expected_output,
                                field_name=field_name, suffix=suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)

        # scenario 2.
        window_size = 3
        step_size = 3
        only_full_windows = False
        input_seqs = np.array([list(range(7)), list(range(2))])
        input_chunk = {field_name: input_seqs}
        expect_seqs = np.array([
            [[0, 1, 2], [3, 4, 5], [6]],
            [[0, 1]]])
        expected_output = {field_name: input_seqs, new_field_name: expect_seqs}
        self._test_window_setup(input_chunk, expected_output,
                                field_name=field_name, suffix=suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)

        # scenario 3.
        window_size = 3
        step_size = 10
        only_full_windows = False
        input_seqs = np.array([list(range(3)), list(range(2))])
        input_chunk = {field_name: input_seqs}
        expect_seqs = np.empty(2, dtype="object")
        expect_seqs[0] = [[0, 1, 2]]
        expect_seqs[1] = [[0, 1]]
        expected_output = {field_name: input_seqs, new_field_name: expect_seqs}

        self._test_window_setup(input_chunk, expected_output,
                                field_name=field_name, suffix=suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)

        # scenario 4.
        window_size = 2
        step_size = 1
        only_full_windows = True
        input_seqs = np.array([list(range(6)), list(range(3)), list(range(1))])
        input_chunk = {field_name: input_seqs}
        expect_seqs = np.array([
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            [[0, 1], [1, 2]],
            []
        ])
        expected_output = {field_name: input_seqs, new_field_name: expect_seqs}

        self._test_window_setup(input_chunk, expected_output,
                                field_name=field_name, suffix=suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)

    def _test_window_setup(self, input_chunk, expected_output_chunk,
                           field_name, suffix,
                           window_size, step_size,
                           only_full_windows):
        window_slider = WindowSlider(field_names=field_name,
                                     window_size=window_size,
                                     step_size=step_size,
                                     new_window_field_name_suffix=suffix,
                                     only_full_windows=only_full_windows)
        actual_output_chunk = window_slider(input_chunk)

        self.assertTrue(equal_data_chunks(expected_output_chunk,
                                          actual_output_chunk))

    def test_tokenizer(self):
        """"""
        field_name = "dummy"
        special_token = "<ANIMAL>"
        lower_case = True
        token_matching_func = lambda x: special_token if x in ["dog", "puppy"]\
            else False
        token_cleaning_func = lambda x: re.sub(r'[?!,.]', '', x)
        tokenization_func = lambda x: x.split()

        input_seqs = np.array(["Hello, this is my dog!",
                               "A dummy sentence for tokenization.",
                               "What a lovely puppy!"])
        input_data_chunk = {field_name: input_seqs}
        expect_seqs = np.array([["hello", "this", "is", "my", special_token],
                                ["a", "dummy", "sentence", "for", "tokenization"],
                                ["what", "a", "lovely", special_token]])
        expected_data_chunk = {field_name: expect_seqs}

        tokenizer = TokenProcessor(field_name, tokenization_func=tokenization_func,
                                   token_cleaning_func=token_cleaning_func,
                                   token_matching_func=token_matching_func,
                                   lower_case=lower_case)
        actual_data_chunk = tokenizer(input_data_chunk)
        self.assertTrue(equal_data_chunks(expected_data_chunk,
                                          actual_data_chunk))


if __name__ == '__main__':
    unittest.main()
