from multiprocessing import Process, Queue
from steps.readers.base_reader import BaseReader
from steps.base_step import BaseStep
from steps.preprocessing.base_preprocessor import BasePreProcessor
from steps.formatters.base_formatter import BaseFormatter
from steps.general.chunk_size_adjuster import ChunkSizeAdjuster
from utils.constants import TERMINATION_TOKEN
from utils.util_funcs.validation import equal_to_constant
from utils.util_funcs.signature_scrapping import scrape_signature
from utils.util_funcs.multi_processing import create_iter_from_queue
from utils.util_funcs.logging_utils import function_logging_decorator
from utils.util_classes import OrderedAttrs
from utils.util_funcs.formatting import format_to_standard_msg_str,\
    format_title
import logging

logger = logging.getLogger("pipeline")


class Pipeline(OrderedAttrs):
    """
    The class is responsible for collecting data-chunks from a reader and
    passing them along the processing pipeline and applying processing
    components/steps in a pre-specified sequential order.

    The pipeline is designed to be used for computationally intensive real-time
    processing of data from arbitrary locations and formats. One also can easily
    take advantage of the parallel CPU architecture.

    Example pipeline:
                  [0. reader] ->
                  [1. log scaling of features (transformer)] ->
                  [2. mapping of categorical features to ids (transformer)] ->
                  [3. to pandas data-frames (formatter)]

    The pipeline will start by reading raw data (e.g. from local csv files) via
    the reader(0.), apply the chain of steps(1., 2.) to data-chunks,
    and finally produce pandas data-frames via the formatter (3.).

    The pipeline has an intermediate format of data-chunks that are passed along
    , which is a dictionary of numpy arrays. The restriction allows to re-use
    pre-implemented steps. However, the user can use a formatter to convert
    data-chunks to a desired format in the end of the pipeline
    (e.g. to pandas data frames). The formatter must be the last step, as it
    outputs data-chunks of arbitrary format.

    The pipeline has three architectures which are created depending on
    worker_processes_number. Each are more suitable for different cases.

    1. worker_processes_number == 0:
        the whole logic is executed on the main processes (both the reader and
        processing steps). The data-chunks are read and processed on demand.

        Most suited for predictions on very small batches or real-time due to
        low overhead.

    2. worker_processes_number == 1:
       A separate process is spawned, and the reader and processing steps
       are assigned to it. Will populate a buffer queue with processed
       data_chunks in a 'lazy' way.

       Well suited for cases when execution of processing steps on data-chunks
       is not very time consuming. However, one does not want to wait for
       processed data-chunks to be created on request.

    3. worker_processes_number > 1:
        A separate process is spawn for the reader, and the rest are spawned
        for data-chunk processing. Namely, each remaining process will have an
        independent copy of the processing steps pipeline. The reader process
        will store raw data-chunks into an input buffer queue from which the
        remaining processes will feed. The processed data-chunks will be stored
        to the output buffer queue.

        Well suited for cases when the processing steps execution on raw
        data-chunks is expensive (e.g. steps contain for-loops), and out-weight
        significantly the reading time of raw data-chunks. In addition,
        it's more beneficial if the hardware is highly parallel (multi-CPU,
        multi-core).

        The downside is that two processing queues are used, and that introduces
        an extra overhead due to serialization and de-serialization, which in
        some cases can out-weight the multi-processing benefits.
    """

    def __init__(self, reader, preprocessor=None,
                 worker_processes_num=0, input_buffer_size=5,
                 output_buffer_size=5, name_prefix=""):
        """
        :param reader: the reader object(subclass of BaseReader) that serves
                       raw data-chunks from a source.
        :param preprocessor: an object(subclass of BasePreprocessor) that
                             contains logic to be executed before data processing
                             starts. For example, it might download data to the
                             local storage, or shuffle data.
        :param worker_processes_num: the number of processes that should be
                                        spawned to read and process data-chunk.
                                        Will affect the architecture of the
                                        pipeline. See the class docstring for
                                        more info.
        :param input_buffer_size: the maximum number of raw data-chunks that are
                                  buffered from the reader
                                  (which runs on a separate process).
                                  Only used when worker_processes_number > 1
        :param output_buffer_size: the maximum number of processed data-chunks
                                   to be accumulated before freezing processing
                                   workers. Only used when
                                   worker_processes_number >= 1.
        :param name_prefix: will be added to msg title in  __str__ out if
                            provided.
        """
        if worker_processes_num < 0:
            raise ValueError("worker_processes_num must be a "
                             "non-negative integer.")

        mess_template = "The provided %s is not valid, as it's not the %s's" \
                        " subclass."
        if not isinstance(reader, BaseReader):
            raise ValueError(mess_template % "reader", BaseReader.__name__)
        if preprocessor is not None and not isinstance(preprocessor,
                                                       BasePreProcessor):
            raise ValueError(mess_template % "preprocessor",
                             BasePreProcessor.__name__)

        super(Pipeline, self).__init__()
        self.worker_processes_number = worker_processes_num
        self.preprocessor = preprocessor
        self.input_buffer_size = input_buffer_size
        self.output_buffer_size = output_buffer_size
        self.name_prefix = name_prefix
        self.reader = reader
        self.steps = []

    def add_step(self, step):
        """
        Adds a new processing component to the end of the chain of processing
        steps.

        :param step: transformer, formatter, or batcher.
        """
        if not isinstance(step, BaseStep):
            raise ValueError(
                "The passed step is invalid. It must either be a"
                " transformer(subclass of BaseTransformer),"
                " formatter(subclass of BaseFormatter), or a batcher."
            )
        # check if the previous step is a formatter, and prevent adding any
        # other steps, as a formatter must be the last step in the chain of
        # steps
        if len(self.steps) > 0 and isinstance(self.steps[-1], BaseFormatter):
            raise ValueError("Can't add a new step because the last one is"
                             " a formatter step, which must be the last one in"
                             " the sequence.")
        self.steps.append(step)

    @function_logging_decorator(logger, log_args=False,
                                class_name="Pipeline")
    def iter(self, **kwargs):
        """
        Creates a generator over processed data-chunks that a user can access.
        The way data-chunks are generated/processed depends on the
        workers_processes_number, see the class's docstring.

        :param kwargs: params to be passed to pre_processor(opt) or reader.
        :return: generator over processed data-chunks.
        """
        if self.preprocessor is not None:
            logger.info("Running the preprocessor")
            kwargs = self.preprocessor(**kwargs)
            logger.info("Done")

        reader_gen = self.reader.iter(**kwargs)
        return self._create_processed_data_chunks_gen(reader_gen)

    def _create_processed_data_chunks_gen(self, reader_gen):
        """
        :return: generator over processed data-chunks.
        """
        if self.worker_processes_number == 0:
            itr = self._create_single_process_gen(reader_gen)
        else:
            itr = self._create_multi_process_gen(reader_gen)
        return itr

    def _create_single_process_gen(self, data_producer):
        """
        Chains reader and steps together, such that the data processing would be
        performed on the main process.

        :return: generator over processed data-chunks.
        """
        return combine_steps_into_chain(data_producer=data_producer,
                                        processing_steps=self.steps)

    def _create_multi_process_gen(self, reader_gen):
        """
        Chains reader and steps together, such that the data processing would
        performed on multiple processes.

        It will create different architectures for data reading and processing
        depending on worker_processes_number. See the class doc string for
        more info.
        """
        term_tokens_received = 0
        output_queue = Queue(self.output_buffer_size)
        workers = []

        if self.worker_processes_number > 1:
            term_tokens_expected = self.worker_processes_number - 1
            input_queue = Queue(self.input_buffer_size)
            reader_worker = _ParallelWorker(reader_gen, input_queue)
            workers.append(reader_worker)

            # adding workers that will process the data
            for _ in range(self.worker_processes_number - 1):
                # since data-chunks will appear in the queue, making an iterable
                # object over it
                queue_iter = create_iter_from_queue(input_queue,
                                                    TERMINATION_TOKEN)
                data_itr = combine_steps_into_chain(data_producer=queue_iter,
                                                    processing_steps=self.steps)
                proc_worker = _ParallelWorker(data_chunk_iter=data_itr,
                                              queue=output_queue)
                workers.append(proc_worker)
        else:
            term_tokens_expected = 1
            data_itr = combine_steps_into_chain(data_producer=reader_gen,
                                                processing_steps=self.steps)
            proc_worker = _ParallelWorker(data_chunk_iter=data_itr,
                                          queue=output_queue)
            workers.append(proc_worker)

        for pr in workers:
            pr.daemon = True
            pr.start()

        while True:
            data_chunk = output_queue.get()
            if equal_to_constant(data_chunk, TERMINATION_TOKEN):
                term_tokens_received += 1
                # need to received all tokens in order to be sure that
                # all data has been processed
                if term_tokens_received == term_tokens_expected:
                    for pr in workers:
                        pr.join()
                    break
                continue
            yield data_chunk

    def __str__(self):
        """Converts the setup/configuration into a human readable string."""
        parent_title, parent_dict = self.get_signature()

        chain = []
        if self.preprocessor:
            chain.append(self.preprocessor)
        chain += self.steps

        children_titles = []
        children_dicts = []
        for step in chain:
            title, attrs = step.get_signature()
            children_titles.append(title)
            children_dicts.append(attrs)

        msg_doc = format_to_standard_msg_str(parent_title=parent_title,
                                             parent_dict=parent_dict,
                                             children_titles=children_titles,
                                             children_dicts=children_dicts)
        return msg_doc

    def get_signature(self):
        """
        Returns the formatted title of the object, and attributes (names and
        values) defining the object as a signature, used for logging and
        printing purposes.

        :return: title(str) and dict of key:value pairs.
        """
        base_title = "%s's SETUP" % self.__class__.__name__
        title = format_title(base_title.upper(),
                             name_prefix=self.name_prefix.upper(),
                             capitalize_prefix=False)

        exl_attrs = ['steps', "name_prefix", "preprocessor"]

        # exclude attrs which are not used for a specific architecture
        if self.worker_processes_number == 0:
            exl_attrs.append('input_buffer_size')
            exl_attrs.append('output_buffer_size')
        if self.worker_processes_number == 1:
            exl_attrs.append('input_buffer_size')

        attrs = scrape_signature(self, excl_attr_names=exl_attrs,
                                 scrape_obj_vals=False)

        return title, attrs


class _ParallelWorker(Process):
    """Worker to execute data reading or processing on a separate process."""

    def __init__(self, data_chunk_iter, queue):
        super(_ParallelWorker, self).__init__()
        self._data_chunk_iterable = data_chunk_iter
        self._queue = queue

    def run(self):
        for data_chunk in self._data_chunk_iterable:
            self._queue.put(data_chunk)
        self._queue.put(TERMINATION_TOKEN)


def combine_steps_into_chain(data_producer, processing_steps):
    """
    Chains processing components/processing_steps sequentially together.
    The chain can be iterated over to produce data-chunks.

    Can be used to produce data-chunks on demand (in the iteration loop).
    Alternatively, can be placed on a separate process that pre-populates a
    queue with processed data-chunks.

    :param data_producer: raw data-chunk producer. E.g. a reader object or just
                          an iterator over raw data-chunks.
    :param processing_steps: list of steps that need to be applied to data-chunks
                             to process them.
    :return: generator.
    """
    prev_step = data_producer
    for new_step in processing_steps:
        if isinstance(new_step, ChunkSizeAdjuster):
            prev_step = new_step.iter(prev_step)
        else:
            prev_step = chain_two_steps(prev_step, new_step)
    return prev_step


def chain_two_steps(data_chunk_iterable, new_step):
    for data_chunk in data_chunk_iterable:
        yield new_step(data_chunk)
