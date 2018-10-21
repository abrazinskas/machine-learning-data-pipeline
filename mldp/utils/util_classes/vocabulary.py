from mldp.utils.util_funcs.paths_and_files import\
    create_file_folders_if_not_exist
from mldp.utils.util_funcs.general import sort_hash, flatten
from mldp.utils.util_classes.ordered_attrs import OrderedAttrs
from mldp.utils.util_funcs.formatting import format_signature, format_title
from mldp.utils.util_funcs.general import listify
from mldp.utils.util_funcs.signature_scrapping import scrape_signature
from mldp.utils.util_funcs.validation import validate_field_names
import itertools
import logging
import codecs
import os
import numpy as np
try:
    import re2 as re
except ImportError:
    import re
logger = logging.getLogger("vocabulary")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEFAULT_SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN}


class _Symbol:
    def __init__(self, token, id, count):
        self.token = token
        self.id = id
        self.count = count


# TODO: it's suboptimal to iterate over datasets multiple times when a number of
# TODO: of vocabularies is created.
# TODO: Implement a function that allows to create multiple vocabularies by
# TODO: iterating over the data only once.

class Vocabulary(OrderedAttrs):
    """
    A general purpose vocabulary class that maps strings(tokens) to symbol
    objects that contain both ids and frequency.

    The vocabulary is populated with symbols, which are objects with:
    tokens, ids, and count attributes.
-
    Operates based on data-chunk iterables, such as readers or data processing
    pipelines.
    """

    def __init__(self, data_chunk_iterable=None, min_count=1, max_size=None,
                 sep=" ", encoding='utf-8', add_default_special_symbols=True,
                 name_prefix=""):
        """
        Notice that default special symbols are not added in the constructor.
        So without running load or create, the vocabulary is empty.

        :param data_chunk_iterable: self-explanatory, do not provide if not
                                    planning to create vocabs.
        :param min_count: minimum frequency of a token to be added.
        :param max_size: maximum size of the vocabulary, tokens that don't fit
                         are discarded.
        :param sep: separator for {token}\sep{count} for reading and writing
                    vocabularies.
        :param encoding: encoding of files that should be used for creation
                         and loading.
        :param add_default_special_symbols: whether default symbols,
                                            such as PAD and UNK should be added.
                                            In some cases, e.g. labels vocab
                                            those symbols are not necessary.
        :param name_prefix: will be added to msg title in  __str__ out if
                            provided.
        """
        if data_chunk_iterable is not None and \
                not hasattr(data_chunk_iterable, "iter"):
            raise ValueError("Please pass a valid data chunk iterable. It has"
                             " to have the iter() method.")

        super(Vocabulary, self).__init__()
        self._data_chunk_iterable = data_chunk_iterable
        self.min_count = min_count
        self.max_size = max_size
        self.sep = sep
        self.encoding = encoding
        self.add_default_special_symbols = add_default_special_symbols
        self.name_prefix = name_prefix
        # below are internal vars
        self._total_count = 0
        self._id_to_symbol = []
        self._token_to_symbol = {}
        self.special_symbols = {}

    def load_or_create(self, vocab_file_path, data_source, data_field_names):
        """
        A convenience function that either creates a vocabulary or loads it if
        it already exists.
        """
        if os.path.isfile(vocab_file_path):
            self.load(vocab_file_path)
        else:
            self.create(data_source=data_source,
                        data_field_names=data_field_names)
            self.write(vocab_file_path)

    def load(self, vocab_file_path):
        """Loads vocabulary from a saved file."""
        with codecs.open(vocab_file_path, encoding=self.encoding) as f:
            for entry in itertools.islice(f, 0, self.max_size):
                # NOTE: I use rstrip below because it causes issues when vocab
                # entry has an empty space as character.
                fields = entry.rstrip().split(self.sep)
                if len(fields) > 2:
                    raise ValueError("The file: '%s' has an incorrect"
                                     " structure, as one or more entries have "
                                     "more than two attributes. It must be in"
                                     " the format {token}{sep}{count}. "
                                     "Alternatively the passed separator '%s'"
                                     " is wrong." %
                                     vocab_file_path, self.sep)
                token, count = fields[0], int(fields[1])

                if count >= self.min_count:
                    symbol = self._add_symbol(token, count=count)
                    if match_special_symbol(token):
                        self.special_symbols[token] = symbol

        if self.add_default_special_symbols:
            self._add_special_symbols(DEFAULT_SPECIAL_TOKENS)

    def create(self, data_source, data_field_names):
        """
        Create vocabulary by passing data_source to the corresponding data-chunk
        iterable and fetching chunks out of it.

        Assumes that tokens are strings, if they are not, it will try to convert
        them to strings.

        :param data_source: dictionary of attributes that should be passed to
                            the data_chunk_iterable.
        :param data_field_names: String or List of (string) attributes that map
                                to the symbols which should be used to create
                                the vocabulary.
        """
        logger.info("Creating a vocabulary for data_attributes: '%s'" %
                    data_field_names)
        try:
            validate_field_names(data_field_names)
        except StandardError as e:
            raise e

        data_field_names = listify(data_field_names)
        temp_token_to_count = {}
        for data_chunk in self._data_chunk_iterable.iter(**data_source):
            for data_attr in data_field_names:
                for tokens in data_chunk[data_attr]:

                    if not isinstance(tokens, (list, np.ndarray)):
                        tokens = [tokens]

                    for token in flatten(tokens):
                        if token == '':
                            continue

                        if not isinstance(token, (int, float, unicode, str)):
                            raise TypeError("Token is not of a correct type"
                                            " (should be int, float, str,"
                                            " unicode)")

                        if isinstance(token, (int, float)):
                            token = str(token)

                        if token not in temp_token_to_count:
                            temp_token_to_count[token] = 0
                        temp_token_to_count[token] += 1

        # populate the collectors
        for token, count in sort_hash(temp_token_to_count, by_key=False):
            if self.max_size and len(self) >= self.max_size:
                break
            if count >= self.min_count:
                symbol = self._add_symbol(token, count)
                self._total_count += count
                if match_special_symbol(token):
                    self.special_symbols[token] = symbol
        if self.add_default_special_symbols:
            self._add_special_symbols(DEFAULT_SPECIAL_TOKENS)

    def write(self, file_path):
        """
        Writes the vocabulary to a plain text file where each line is of the
        form: {token}{sep}{count}.
        """
        create_file_folders_if_not_exist(file_path)
        with codecs.open(file_path, 'w', encoding=self.encoding) as f:
            for symbol in self:
                if self.add_default_special_symbols and symbol.token in\
                        DEFAULT_SPECIAL_TOKENS:
                    continue
                try:
                    str_entry = self.sep.join([symbol.token, str(symbol.count)])
                    f.write(str_entry + "\n")
                except StandardError:
                    logger.fatal(
                        "Below entry produced a fatal error in write().")
                    logger.fatal(symbol.token)
                    raise ValueError("Could not process a token")
        logger.info("Vocabulary written to: '%s'." % file_path)

    def get_signature(self):
        """
        Returns the formatted title of the object, and attributes (names and
        values) defining the object as a signature, used for logging and
        printing purposes.

        :return: title(str) and dict of key:value pairs.
        """
        title = format_title(self.__class__.__name__, self.name_prefix)
        excl_attrs = ["_id_to_symbol", "_token_to_symbol", "name_prefix"]
        signature_attrs = scrape_signature(self, excl_attrs)
        signature_attrs["vocab_size"] = len(self)
        return title, signature_attrs

    def __str__(self):
        """Converts the setup/configuration into a human readable string."""
        title, signature_attrs = self.get_signature()
        my_str = format_signature(title, attrs=signature_attrs, indent=2)
        return my_str

    def __len__(self):
        return len(self._id_to_symbol)

    def __contains__(self, entry):
        """
        A generic containment function that assumes item to be either int, str,
        or a symbol object.
        """
        if isinstance(entry, _Symbol):
            return entry.token in self._token_to_symbol
        if isinstance(entry, str) or isinstance(entry, unicode):
            return entry in self._token_to_symbol
        if isinstance(entry, int):
            return len(self._id_to_symbol) > entry
        raise ValueError('Input argument is not of a correct type.')

    def __iter__(self):
        for symbol in self._id_to_symbol:
            yield symbol

    def __getitem__(self, entry):
        """
        A generic get method that will return <UNK> special symbol object if
        word(s) are not in the vocabulary (unless the special symbol does not
        exist).

        :param entry: either an int (id) or string (token) or a list of
                      str (tokens).
        :return: symbol object.
        """
        if isinstance(entry, str) or isinstance(entry, unicode):
            if entry in self:
                return self._token_to_symbol[entry]
            else:
                if UNK_TOKEN in self:
                    return self[UNK_TOKEN]
                else:
                    raise ValueError(
                        "Item %s is not present in the vocabulary" % (
                            str(entry)))
        if isinstance(entry, (int, long, np.integer)):
            return self._id_to_symbol[entry]
        if isinstance(entry, (list, np.ndarray)):
            return [self[w] for w in entry]
        logger.fatal("Below entry produced a fatal error in __getitem__().")
        logger.fatal(type(entry))
        logger.fatal(entry)
        raise ValueError('Input argument is not of a correct type.')

    def _add_symbol(self, token, count=1):
        """
        Adds an entry to the collection or updates its count if it already
        present.

        :return: created or updated symbol object.
        """
        if not isinstance(token, (str, unicode)):
            raise TypeError("Token must be a string or unicode!")
        if token in self:
            symbol = self[token]
            self._total_count += count
            self._total_count -= symbol.count
            symbol.count = count
        else:
            n = len(self._token_to_symbol)
            symbol = _Symbol(token, id=n, count=count)
            self._token_to_symbol[token] = symbol
            self._id_to_symbol.append(symbol)
            self._total_count += count
        return symbol

    def _add_special_symbols(self, special_symbols):
        """Appends or updates special symbols."""
        for token in special_symbols:
            count = self[token].count if token in self else 1
            self.special_symbols[token] = self._add_symbol(token, count)


def match_special_symbol(token):
    """Checks whether the passed token matches the special symbols format."""
    return re.match(r'<[A-Z]+>', token)
