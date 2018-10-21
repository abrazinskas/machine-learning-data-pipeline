from collections import OrderedDict
import types
from general import listify, is_custom_object


def scrape_signature(obj, excl_attr_names=None, excl_types=None,
                     allowed_types=(int, str, unicode, float, list, dict,
                                    OrderedDict, bool,
                                    types.FunctionType, types.MethodType,
                                    types.LambdaType, types.BuiltinFunctionType),
                     scrape_obj_vals=False):
    """
    Extracts attributes from objects that potentially represent the object
    itself. This function is used to create automatic documentation of steps
    in the pipeline.

    Lambda functions are not represented beyond bare 'lambda' strs due to
    complexity of obtaining a better representation.

    :param obj: an object from which to scrape signature.
    :param excl_attr_names: a list of attribute names that should not be
                            scrapped.
    :param excl_types: a list of attr value types, which should not be ignored.
                       Namely, if an attribute value if of the excluded type -
                       the attribute is ignored.
    :param allowed_types: self-explanatory.
    :param scrape_obj_vals: whether to represent values which are custom
                            objects.
    :return: a dictionary of attribute key value pairs.
    """
    if excl_types:
        excl_types_set = set(excl_types)
        allowed_types = tuple([t for t in allowed_types if t not in
                               excl_types_set])
    collector = OrderedDict()
    d = obj.__odict__ if hasattr(obj, "__odict__") else obj.__dict__
    for attr_name, attr_value in d.items():
        excluded_attr = excl_attr_names is not None and \
                        attr_name in excl_attr_names
        protected_attr = attr_name[0] == "_"
        if excluded_attr or protected_attr:
            continue
        try:
            collector[attr_name] = represent_val(attr_value,
                                                 allowed_types=allowed_types,
                                                 excl_attr_names=excl_attr_names,
                                                 scrape_obj_vals=scrape_obj_vals
                                                 )
        except ValueError:
            pass
    return collector


def represent_val(val, allowed_types, excl_attr_names, scrape_obj_vals=False):
    """
    Attempts to represent the value, raises an error if fails.
    Certain value types have their own rules regarding representation.
    """
    error = ValueError("Can't represent the value.")

    if not valid_val_type(val, allowed_types):
        # 0. Custom objects
        if scrape_obj_vals and is_custom_object(val):
            if hasattr(val, 'get_signature'):
                _, attrs = val.get_signature()
            else:
                attrs = scrape_signature(val, excl_attr_names=excl_attr_names,
                                         allowed_types=allowed_types,
                                         scrape_obj_vals=scrape_obj_vals)
            return "%s(%s)" % (val.__class__.__name__, format_dict(attrs))
        else:
            raise error

    # 2. Methods and standard functions
    if is_of_allowed_types(val,
                           target_types=[types.FunctionType, types.MethodType,
                                         types.BuiltinFunctionType,
                                         types.LambdaType],
                           allowed_types=allowed_types):
        return repr_func(val)

    # 3. Lists
    if is_of_allowed_types(val, list, allowed_types=allowed_types):
        res = []
        for el in val:
            try:
                res.append(represent_val(el, allowed_types=allowed_types,
                                         excl_attr_names=excl_attr_names,
                                         scrape_obj_vals=scrape_obj_vals))
            except ValueError:
                pass
        if len(res):
            return "[" + ", ".join(res) + "]"
        raise error

    # 4. Dicts and OrderedDicts
    is_allowed_dict = is_of_allowed_types(val, dict, allowed_types)
    is_allowed_o_dict = is_of_allowed_types(val, OrderedDict, allowed_types)
    if is_allowed_dict or is_allowed_o_dict:
        if not len(val):
            return format_dict(val)

        res = dict() if is_allowed_dict else OrderedDict()
        for k, v in val.items():
            try:
                res[k] = represent_val(v, allowed_types=allowed_types,
                                       excl_attr_names=excl_attr_names,
                                       scrape_obj_vals=scrape_obj_vals)
            except ValueError:
                pass
        if len(res):
            return format_dict(res)
        # if it fails to represent dict values - will return keys only
        return "(keys)%s" % ("{" + ", ".join(val.keys()) + "}")

    return repr(val)


def format_dict(d):
    tmp_k_v_str = [("%s: %s" % (k, v)) for k, v in d.items()]
    return "{" + ", ".join(tmp_k_v_str) + "}"


def valid_val_type(val, allowed_types):
    return isinstance(val, allowed_types) or val is None


def is_of_allowed_types(val, target_types, allowed_types):
    """Checks if the value belongs to a specific target types(if allowed)."""
    target_types = listify(target_types)
    allowed_types = allowed_types
    return any(
        [t in allowed_types and isinstance(val, t) for t in target_types])


def repr_func(func):
    """Attempts to return a representative document of a function/method."""
    try:
        if hasattr(func, "im_self"):
            im_self = func.im_self
            full_class_name = str(im_self.__class__)
            func_name = func.__name__
            return ".".join([full_class_name, func_name])
        else:
            return func.__name__
    except StandardError:
        return str(func)