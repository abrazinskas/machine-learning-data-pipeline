import unittest
from mldp.utils.util_classes import Vocabulary
from mldp.steps.readers import CsvReader
from mldp.utils.util_funcs.paths_and_files import get_file_paths
from common import read_from_csv_file
import numpy as np
import os


class TestVocabularyUtil(unittest.TestCase):

    def setUp(self):
        self.reader = CsvReader()

    def test_creation(self):
        data_path = 'data/small_chunks/'
        data_source = {"data_path": data_path}
        vocab = Vocabulary(self.reader)
        vocab.create(data_source, "first_name")

        data = read_from_csv_file(get_file_paths(data_path))
        unique_first_names = np.unique(data['first_name'])

        for ufn in unique_first_names:
            self.assertTrue(ufn in vocab)

    def test_loading(self):
        tmp_f_path = "dummy_vocab.txt"
        sep = '\t'
        entries = [("first", "1"), ("two", "2"), ("three", "3"), ("four", "4"),
                   ("five", "5"), ("seven", "7")]
        with open(tmp_f_path, 'w') as f:
            for entry in entries:
                f.write(sep.join(entry) + "\n")

        vocab = Vocabulary(min_count=1, sep="\t")
        vocab.load(tmp_f_path)

        for token, count in entries:
            self.assertTrue(token in vocab)
            self.assertTrue(vocab[token].count == int(count))
        os.remove(tmp_f_path)

    def test_when_unk_symbol_is_present(self):
        data_path = 'data/small_chunks/'
        data_source = {"data_path": data_path}
        vocab = Vocabulary(self.reader, add_default_special_symbols=True)
        vocab.create(data_source, "first_name")

        a = vocab["dummy_token"]
        self.assertTrue(True)

    def test_when_unk_symbol_is_absent(self):
        data_path = 'data/small_chunks/'
        data_source = {"data_path": data_path}
        vocab = Vocabulary(self.reader, add_default_special_symbols=False)
        vocab.create(data_source, "first_name")

        with self.assertRaises(StandardError):
            a = vocab["dummy_token"]


if __name__ == '__main__':
    unittest.main()
