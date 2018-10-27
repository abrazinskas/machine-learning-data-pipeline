import numpy as np
import pandas
from mldp.utils.util_classes import DataChunk


def create_list_of_data_chunks(data_chunk, chunk_size):
    """Creates a list of data-chunks out of the passed data-chunk."""
    collector = []
    start_indx = 0
    while start_indx < len(data_chunk):
        slice_range = range(start_indx, min(start_indx + chunk_size,
                                            len(data_chunk)))
        dc = DataChunk({k: v[slice_range] for k, v in data_chunk.items()})
        collector.append(dc)
        start_indx += chunk_size
    return collector


def generate_data_chunk(data_attrs_number, data_size):
    """Generated a data-chunk with random 1D values of data_size."""
    data = {str(i): np.random.rand(data_size) for
            i in range(data_attrs_number)}
    data_chunk = DataChunk(data)
    return data_chunk


def read_data_from_csv_file(file_path, **kwargs):
    if isinstance(file_path, list):
        data = pandas.concat([pandas.read_csv(p, **kwargs) for p in file_path])
    else:
        data = pandas.read_csv(file_path, **kwargs)
    r = data.to_dict('list')

    for k, v in r.items():
        v = np.array(v)
        if v.dtype not in [int, float]:
            v = v.astype(object)
        r[k] = v

    return DataChunk(r)
