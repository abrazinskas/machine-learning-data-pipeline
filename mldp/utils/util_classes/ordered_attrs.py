from collections import OrderedDict


class OrderedAttrs(object):
    """
    Allows to store object attributes in the order of assignment in a sep var.
    Used for automatic information scraping for objects logging.
    """
    def __init__(self):
        self.__odict__ = OrderedDict()

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key != "__odict__":
            self.__odict__[key] = value