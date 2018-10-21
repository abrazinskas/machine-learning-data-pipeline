from mldp.utils.util_classes.ordered_attrs\
    import OrderedAttrs
from mldp.utils.util_funcs.signature_scrapping import\
    scrape_signature
from mldp.utils.util_funcs.formatting import format_signature, format_title


class BaseStep(OrderedAttrs):
    """Base class of all steps used in the Pipeline class."""

    def __init__(self, name_prefix=None):
        """
        :param name_prefix: a str that will prefix the title of the object if
                            the signature is generated.
        """
        super(BaseStep, self).__init__()
        self.name_prefix = name_prefix

    def get_signature(self):
        """
        Returns the formatted title of the object, and attributes (names and
        values) defining the object as a signature, used for logging and
        printing purposes.

        :return: title(str) and dict of key:value pairs.
        """
        title = format_title(self.__class__.__name__,
                             name_prefix=self.name_prefix)
        attrs = scrape_signature(self, excl_attr_names=['name_prefix'],
                                 scrape_obj_vals=True)
        return title, attrs

    def __str__(self):
        title, attrs = self.get_signature()
        return format_signature(title, attrs, indent=2)
