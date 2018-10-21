import logging
import os
from mldp.utils.util_funcs.paths_and_files import safe_mkdir
from time import strftime


def function_logging_decorator(logger, log_args=True, log_kwargs=True,
                               class_name=""):
    """
    Creates/returns a decorator specific to a passed logger. Namely, it logs
    info about the function into via provided logger.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            mess = "Executing %s" % func.__name__
            if class_name != "":
                mess += " of %s" % class_name
            if log_args:
                mess += ", args: %s" % args
            if log_kwargs:
                mess += ", kwargs: %s" % kwargs
            logger.info(mess)
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def init_logger(logger_name=__name__, output_folder=None, level=logging.NOTSET,
                log_file_name="log_" + strftime("%b_%d_%H_%M_%S") + '.txt'):
    """Initializes a standard logger for console and file writing."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

    # adding console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_folder:
        safe_mkdir(output_folder)
        log_file_path = os.path.join(output_folder, log_file_name)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
