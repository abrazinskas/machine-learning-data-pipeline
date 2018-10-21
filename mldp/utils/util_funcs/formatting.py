

def format_big_box(message, ws_offset=20):
    """
    Formats a message by wrapping it into a big box of # and white space
    symbols.

    :param message: self-explanatory.
    :param ws_offset: two sided white space offset wrapping the message.
    """
    total_len = len(message) + ws_offset

    # 2 is added because of # on both sides
    top_or_bottom = '#' * (total_len + 2)
    left_side = ws_offset / 2
    right_side = ws_offset / 2
    if ws_offset % 2 == 1:
        right_side += 1

    center = '#' + ' ' * left_side + message + ' ' * right_side + '#'
    mess = "\n%s\n%s\n%s\n" % (top_or_bottom, center, top_or_bottom)

    return mess


def format_small_box(message, ws_offset=6, box_width=40):
    """
    Formats the message into a one line string offset by # and white space
    symbols.

    :param message: self-explanatory.
    :param ws_offset: two sided white space offset wrapping the message.
    :param box_width: the total width of the box.
    """
    msg = "\n"
    hashes_count = max(box_width - (len(message) + ws_offset), 0)
    left_side = hashes_count/2
    right_side = hashes_count/2
    if hashes_count % 2 == 1:
        right_side += 1
    left_ws_offset = ws_offset/2
    right_ws_offset = ws_offset/2
    if ws_offset % 2 == 1:
        right_ws_offset += 1

    msg += ''.join(['#' * left_side, ' ' * left_ws_offset,
                    message,
                    ' ' * right_ws_offset,
                    '#' * right_side]) + "\n"
    return msg


def format_dict(dic, indent=0):
    """
    Formats dictionary as a string of key value pairs. Each pair is printed on
    a new line.
    """
    msg = ""
    for param_name, param_value in dic.items():
        msg += ' ' * indent + (param_name + ": " + str(param_value) + '\n')
    return msg


def format_to_standard_msg_str(parent_title, parent_dict, parent_ws_offset=40,
                               indent=2, children_titles=None,
                               children_dicts=None, children_ws_offset=6,
                               ):
    """Creates a standard print-out used for automatic objects documentation."""
    msg = ""
    msg += format_big_box(parent_title, ws_offset=parent_ws_offset)
    msg += "\n"
    msg += format_dict(parent_dict, indent=indent)

    parent_box_width = len(parent_title) + parent_ws_offset + 2

    if children_titles and children_dicts:
        assert len(children_titles) == len(children_dicts)
        for title, child_dict in zip(children_titles, children_dicts):
            msg += format_signature(title, attrs=child_dict, indent=indent,
                                    ws_offset=children_ws_offset,
                                    box_width=parent_box_width)
    msg += format_big_box("", ws_offset=len(parent_title) + parent_ws_offset)
    return msg


def format_signature(title, attrs, indent=0, **format_small_box_kwargs):
    """A common formatting for conversion of signature to str."""
    msg = format_small_box(title, **format_small_box_kwargs)
    msg += "\n"
    msg += format_dict(attrs, indent=indent)
    return msg


def format_title(title, name_prefix=None, capitalize_prefix=True):
    if name_prefix:
        if capitalize_prefix:
            name_prefix = name_prefix.capitalize()
        title = "%s %s" % (name_prefix, title)
    return title
