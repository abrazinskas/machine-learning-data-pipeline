from pandas.io.parsers import TextFileReader
from mldp.utils.constants import TERMINATION_TOKEN
from mldp.utils.util_classes import fs_accessor_factory
from mldp.utils.util_funcs.paths_and_files import is_s3_path,\
    filter_file_paths_by_extension
from functools import partial as fun_partial


class TextFileReaderMod(TextFileReader):
    """
    A modified version of the Pandas TextFileReader, which does not convert
    read data into Pandas DataFrames but instead leaves data in the dictionary
    of the numpy arrays format.
    """
    def read(self, nrows=None):
        if nrows is not None:
            if self.options.get('skipfooter'):
                raise ValueError('skipfooter not supported for iteration')
        fields_and_fields_to_data_tuple = self._engine.read(nrows)
        if self.options.get('as_recarray'):
            return fields_and_fields_to_data_tuple
        # May alter columns / col_dict
        index, columns, col_dict = \
            self._create_index(fields_and_fields_to_data_tuple)
        return col_dict


def populate_queue_with_chunks(itr_creator, queue):
    """
    This function is used by thread workers in order to load and store data
    chunks to the common chunk queue.

    :param itr_creator: a function that creates an iterable over data-chunks.
    :param queue: self-explanatory.
    :return: None.
    """
    try:
        it = itr_creator()
        for data_chunk in it:
            queue.put(data_chunk)
    except Exception as e:
        queue.put(e)
        return
    queue.put(TERMINATION_TOKEN)


def create_openers_of_valid_files(paths, ext='.csv'):
    """
    Returns a list of (valid) file paths openers. Concretely, functions on call
    return opened file objects. In such a way we hide the details on how to open
    files of different types (s3 or local atm) and avoid opening all files at
    once.
    """
    valid_file_openers = []
    for path in paths:
        fs = fs_accessor_factory("s3" if is_s3_path(path) else "local")
        file_paths = fs.list_file_paths(path)
        valid_file_paths = filter_file_paths_by_extension(file_paths,
                                                          ext=ext)
        valid_file_openers += [fun_partial(fs.open_file, path=p, mode='r')
                               for p in valid_file_paths]
    return valid_file_openers
