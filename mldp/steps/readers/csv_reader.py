from mldp.utils.constants import TERMINATION_TOKEN
from mldp.steps.readers.base_reader import BaseReader
from mldp.steps.readers.common import TextFileReaderMod, \
    create_openers_of_valid_files
from mldp.utils.util_funcs.validation import validate_data_paths
from functools import partial as fun_partial
from mldp.utils.util_funcs.general import listify
from multiprocessing.dummy import Pool
from Queue import Queue


class CsvReader(BaseReader):
    """
    CSV reader is based on a pandas internal reader (TextReader) for reading of
    raw data-chunks and inferring column types.

    Csv files must have .csv extensions, otherwise files will be ignored. Works
    for s3 and local storage csv files.

    The implementation is multi-threaded, and can assign separate files to be
    read by different threads. However, in that case data-chunks are read
    without the preserved order guarantees. Will not give any advantage if
    the pandas reading engine is not 'c'. Please refer to pandas.read_csv for
    details.

    Finally, if chunk_size requires exceeds a file's data units count, an
    incomplete data-chunk will be produced.
    """

    def __init__(self, chunk_size=1000, worker_threads_num=1,
                 buffer_size=5, name_prefix=None, engine='c', **kwargs):
        """
        :param chunk_size: the intermediate number of data units that are
                           passed along the pipeline. Larger data-chunks consume
                           more memory but can be beneficial for vectorized
                           operations (e.g. np.log).
        :param worker_threads_num: a number of workers that should be assigned
                                   to reading separate files. Will not be give
                                   an advantage if the pandas reading engine is
                                   not 'c'(engine='c').
                                   Please refer to pandas.read_csv for details.
        :param buffer_size: the maximum number of data-chunks that a buffer
                            queue can contain. The collector queue is populated
                            by threads. And threads freeze off-loading chunks
                            if the queue is full. They resume when the queue
                            gets free slots.
        :param engine: whether to use 'c' or 'python' modified pandas csv reader.
        :param kwargs: additional parameters that should be passed to the pandas
                       reader (see pandas.read_csv).
        """
        super(CsvReader, self).__init__(chunk_size=chunk_size,
                                        name_prefix=name_prefix)
        if type(worker_threads_num) != int or worker_threads_num <= 0:
            raise ValueError("Please provide a valid integer for"
                             " worker_threads_number."
                             " It must be non-negative.")
        if worker_threads_num > 1 and engine != 'c':
            raise Warning("There will be no advantage in multi-threading if"
                          " the engine is not set to 'c' due to GIL.")
        if worker_threads_num == 1:
            buffer_size = None

        self.worker_threads_num = worker_threads_num
        self.parser_kwargs = kwargs
        self.buffer_size = buffer_size

    def _iter(self, data_path):
        """
        :param data_path: a string corresponding to a location of data
                          (file or folder). If list is provided, will assume
                          multiple data paths.
                          Will work for local data storage or S3.
                          In the case of s3, it should have the s3://path
                          format.
        :return: iterator (generator) over data-chunks, which are dictionaries
                 of numpy arrays. Namely, each column gets a key and values are
                 stored into numpy arrays.
        """
        try:
            validate_data_paths(data_path)
        except StandardError as e:
            raise e

        data_paths = listify(data_path)
        file_openers = create_openers_of_valid_files(data_paths, ext='.csv')
        if not file_openers:
            raise ValueError(
                "No valid files to open, please check the provided"
                " data_paths(%s). Note that files without %s extension"
                " are ignored." % ('.csv', data_paths))

        if self.worker_threads_num > 1:
            chunk_iter = self._create_multi_th_gen(file_openers)
        else:
            chunk_iter = self._create_single_th_iter(file_openers)
        return chunk_iter

    def _create_multi_th_gen(self, file_openers):
        """
        Creates a multi-threaded generator over data-chunks by reading data from
        data_paths. Works by spawning thread workers and assigning csv files to
        them. Threads populate a queue with raw data-chunks.

        :param file_openers: a list of function that return opened files
        """
        # the queue will accumulate raw data-chunks produced by threads
        chunk_queue = Queue(maxsize=self.buffer_size)
        # function's partial that only will expect a file opener used by workers
        func = fun_partial(self.get_data_chunk_iter,
                           chunk_size=self.chunk_size,
                           iterable=True,
                           queue=chunk_queue,
                           parser_kwargs=self.parser_kwargs)

        # creating a pool of threads, and assigning jobs to them
        pool = Pool(self.worker_threads_num)
        pool.map_async(func, file_openers)
        pool.close()  # indicate that never going to submit more work

        # the inf. while loop is broken when all files are read,
        # i.e. a termination token is received for each file
        received_termin_tokens_count = 0
        while True:
            chunk = chunk_queue.get(timeout=5)
            if isinstance(chunk, Exception):
                raise chunk
            if chunk == TERMINATION_TOKEN:
                received_termin_tokens_count += 1
                if received_termin_tokens_count == len(file_openers):
                    pool.join()
                    break
            else:
                yield chunk

    def _create_single_th_iter(self, file_openers, **parser_kwargs):
        """Single threaded generator that avoid using Pools and Queues."""
        for file_opener in file_openers:
            dc_iter = self.get_data_chunk_iter(file_opener,
                                               chunk_size=self.chunk_size,
                                               **parser_kwargs)
            for chunk in dc_iter:
                yield chunk

    @staticmethod
    def get_data_chunk_iter(file_opener, chunk_size, **parser_kwargs):
        """
        Create and return a modified pandas data generator that spits out
        dictionaries of numpy arrays, instead of data-frames.
        """
        pandas_iter = TextFileReaderMod(file_opener(),
                                        chunksize=chunk_size,
                                        iterable=True,
                                        **parser_kwargs)
        return pandas_iter
