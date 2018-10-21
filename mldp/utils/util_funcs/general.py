from collections import Iterable
import operator


def is_custom_object(obj):
    return hasattr(obj, '__dict__') or hasattr(obj, '__slots__')


def all_elements_are_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def flatten(my_list):
    """Flattens nested lists ."""
    curr_items = []
    for x in my_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, unicode, bytes)):
            curr_items += flatten(x)
        else:
            curr_items.append(x)
    return curr_items


def sort_hash(hash, by_key=True, reverse=True):
    if by_key:
        indx = 0
    else:
        indx = 1
    return sorted(hash.items(), key=operator.itemgetter(indx), reverse=reverse)


def listify(val):
    """If val is an element the func wraps it into a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    return [val]
