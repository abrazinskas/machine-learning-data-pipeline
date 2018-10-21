import os
import re
import errno


def get_file_paths(dir_path):
    """
    :param dir_path: self-explanatory.
    :return: a list of file paths that are in the folder. If dir_path is
             actually a file_path it will be returned in the list.
    """
    if not os.path.exists(dir_path):
        raise ValueError("The path '%s' does not exist!" % dir_path)
    if os.path.isdir(dir_path):
        paths = []
        for f_name in os.listdir(dir_path):
            f_path = os.path.join(dir_path, f_name)
            if os.path.isfile(f_path):
                paths.append(f_path)
    else:
        paths = [dir_path]
    return paths


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_s3_path(path):
    if re.match(r'^s3:\/\/.*$', path):
        return True
    else:
        return False


def filter_file_paths_by_extension(file_paths, ext='.csv'):
    """
    Filters out file paths that do not have an appropriate extension.

    :param file_paths: list of file path strings
    :param ext: valid extension
    """
    valid_file_paths = []
    for file_path in file_paths:
        if not file_path.endswith(ext):
            continue
        valid_file_paths.append(file_path)
    return valid_file_paths


def create_file_folders_if_not_exist(file_path):
    """Creates folders associated with host of the file."""
    if os.path.dirname(file_path) and not\
            os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise