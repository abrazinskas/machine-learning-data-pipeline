from base_transformer import BaseTransformer
from mldp.utils.util_funcs.validation import validate_field_names
from mldp.utils.util_funcs.nlp.sequences import pad_sequences as ps
from common import create_new_field_name
from mldp.utils.util_funcs.general import listify
import numpy as np


class Padder(BaseTransformer):
    """
    This transformer pads sequences with a special pad_symbol to assure that
    all sequences are of the same length.

    Creates a separate field for each field that is padded.

    This transformer is useful if a model expects tensors as input.

    Works only for 2D data (i.e. batch_size x 1D_sequences)
    and 3D data (i.e. batch_size x num_seqs x 1D_sequences).
    """

    def __init__(self, field_names, pad_symbol, symbol_to_mask=None,
                 padding_mode='both', axis=1,
                 new_mask_field_name_suffix="mask", **kwargs):
        """
        :param field_names: str or list of str names that should represent
                            fields that should be padded.
        :param pad_symbol: a symbol that should be used for padding.
        :param symbol_to_mask: a symbol(token) that should be masked in
                               sequences. E.g. Can be used to mask <UNK> tokens.
        :param padding_mode: left, right, or both. Defines the side to which
                             padding symbols should be appended.
        :param axis: defines an axis of data to which padding should be applied.
                     Currently only axes 1 or 2 are supported.
        :param new_mask_field_name_suffix: a suffix of a new padded field that is
                                          created for each field_names.
                                          See create_mask_field_name().
        """
        try:
            validate_field_names(field_names)
        except StandardError as e:
            raise e

        super(Padder, self).__init__(**kwargs)
        self.field_names = listify(field_names)
        self.pad_symbol = pad_symbol
        self.symbol_to_mask = symbol_to_mask
        self.padding_mode = padding_mode
        self.axis = axis
        self.new_mask_fn_suffix = new_mask_field_name_suffix

    def _transform(self, data_chunk):
        for fn in self.field_names:
            fv = data_chunk[fn]
            mask_fn = create_new_field_name(fn, suffix=self.new_mask_fn_suffix)
            if self.axis == 1:
                padded_seqs, mask = ps(fv,
                                       pad_symbol=self.pad_symbol,
                                       mask_present_symbol=self.symbol_to_mask,
                                       padding_mode=self.padding_mode)
            else:
                padded_seqs = np.empty(len(fv), dtype="object")
                mask = np.empty(len(fv), dtype="object")
                for i, el in enumerate(fv):
                    c_pad_seqs, c_mask = ps(el,
                                            pad_symbol=self.pad_symbol,
                                            mask_present_symbol=self.symbol_to_mask,
                                            padding_mode=self.padding_mode
                                            )
                    padded_seqs[i] = c_pad_seqs
                    mask[i] = c_mask
            data_chunk[fn] = padded_seqs
            data_chunk[mask_fn] = mask
        return data_chunk
