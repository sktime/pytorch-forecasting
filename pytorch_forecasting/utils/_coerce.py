"""Coercion functions for various data types."""

from copy import deepcopy


def _coerce_to_list(obj):
    """Coerce object to list.

    None is coerced to empty list, otherwise list constructor is used.
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    return list(obj)


def _coerce_to_dict(obj):
    """Coerce object to dict.

    None is coerce to empty dict, otherwise deepcopy is used.
    """
    if obj is None:
        return {}
    return deepcopy(obj)
