from mldp.steps.transformers.base_transformer import BaseTransformer
from mldp.utils.util_funcs.validation import validate_field_names
from mldp.utils.util_funcs.general import listify


class FieldsSelector(BaseTransformer):
    """
    Selects features/fields specified in attr_names from data-chunks and
    drops the rest.
    """

    def __init__(self, field_names, **kwargs):
        """
        :param field_names: str or list of str names that should represent
                            fields that should be selected from data-chunks.
                            Other fields are discarded.
        """
        try:
            validate_field_names(field_names)
        except StandardError as e:
            raise e

        super(FieldsSelector, self).__init__(**kwargs)
        self.field_names = listify(field_names)

    def _transform(self, data_chunk):
        return {name: data_chunk[name] for name in self.field_names}
