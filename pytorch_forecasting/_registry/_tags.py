"""Register of object tags.

This module exports the following:

OBJECT_TAG_REGISTER - list of tuples
    each tuple corresponds to a tag, elements as follows:
        0 : string - name of the tag as used in the ``_tags`` dictionary
        1 : string - name of the object type the tag applies to
        2 : string or tuple - expected type of the tag value
            if string, one of "bool", "int", "str"
            if tuple, ("str", list_of_str) or ("list", list_of_str) or
            ("list", "str"); for the first two the value must be one of, or a
            sublist of, the second element
        3 : string - plain English description of the tag

OBJECT_TAG_TABLE - pd.DataFrame
    OBJECT_TAG_REGISTER in table form, columns 0 to 3 as above

OBJECT_TAG_LIST - list of string
    elements are 0-th entries of OBJECT_TAG_REGISTER, in same order
"""

# based on the sktime and skpro modules of same name

__author__ = ["echo-xiao"]

import inspect
import sys

import pandas as pd

from pytorch_forecasting.base._base_object import _BaseObject


class _BaseTag(_BaseObject):
    """Base class for all tags."""

    _tags = {
        "object_type": "tag",
        "tag_name": "",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "",
    }


# --------------------------
# Object identification
# --------------------------


class object_type(_BaseTag):
    """Scitype of the object, used to dispatch lookup and testing.

    Possible values
    ---------------
    "forecaster_pytorch"       alias carried by every v1 forecaster package
    "forecaster_pytorch_v1"    v1 forecaster package, a ``_pkg`` class
    "forecaster_pytorch_v2"    v2 forecaster package, a ``_pkg_v2`` class
    "metric"                   metric package
    "scaler_strategy"          scaler strategy used by ``ScalerAdapter``
    "tag"                      a tag class in this module

    The value is a ``str`` on most base classes and a ``list`` of ``str`` on
    ``_BasePtForecaster``, which carries both "forecaster_pytorch" and
    "forecaster_pytorch_v1". Both forms are accepted.

    Default
    -------
    Inherited from the base class the object descends from. Concrete packages
    do not set it themselves.

    Effect
    ------
    ``all_objects`` filters on it, and ``get_test_class_registry`` in
    ``tests/test_class_register.py`` maps "forecaster_pytorch_v1" and
    "forecaster_pytorch_v2" to the test class run against the object.
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": (
            "list",
            [
                "forecaster_pytorch",
                "forecaster_pytorch_v1",
                "forecaster_pytorch_v2",
                "metric",
                "scaler_strategy",
                "tag",
            ],
        ),
        "short_descr": "scitype of the object, e.g. 'metric'",
    }


# --------------------------
# Metrics
# --------------------------


class metric_type(_BaseTag):
    """Kind of prediction the metric scores.

    Possible values
    ---------------
    "point"                 point forecasts, ``y_pred`` of shape (batch, time)
    "quantile"              quantile forecasts, ``y_pred`` carries a trailing
                            quantile dimension
    "distribution"          distributional forecasts, ``y_pred`` holds
                            distribution parameters
    "point_classification"  classification over discrete target values

    Default
    -------
    No default. Metric packages must declare this tag.

    Effect
    ------
    Read by ``TestAllMetrics`` in ``metrics/tests/test_all_metrics.py``:
    "quantile" makes the test harness add a quantile dimension to ``y_pred``
    before scoring, and "quantile" and "point_classification" change which
    reduction assertions run.
    """

    _tags = {
        "tag_name": "metric_type",
        "parent_type": "metric",
        "tag_type": (
            "str",
            ["point", "quantile", "distribution", "point_classification"],
        ),
        "short_descr": "kind of prediction the metric scores",
    }


# --------------------------
# Forecaster capabilities
# --------------------------


class capability__exogenous(_BaseTag):
    """Whether the model can use exogenous covariates.

    Possible values
    ---------------
    True   the model uses exogenous variables in a non-trivial way
    False  the model ignores exogenous inputs

    Default
    -------
    No default. An undeclared tag reads back as ``None``, not ``False``.

    Effect
    ------
    Read by the ``model_overview`` sphinx extension
    (``docs/source/_ext/model_overview.py``), which fills the "Covariates"
    column of the generated model overview table. No runtime code branches
    on it.
    """

    _tags = {
        "tag_name": "capability:exogenous",
        "parent_type": "forecaster_pytorch",
        "tag_type": "bool",
        "short_descr": "whether the model uses exogenous covariates",
    }


OBJECT_TAG_REGISTER = []

for _, _cl in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if _cl is _BaseTag or not issubclass(_cl, _BaseTag):
        continue

    _cl_tags = _cl.get_class_tags()
    _tag_name = _cl_tags["tag_name"]
    _parent_type = _cl_tags["parent_type"]
    _tag_type = _cl_tags["tag_type"]
    _short_descr = _cl_tags["short_descr"]

    if isinstance(_parent_type, list):
        for _p_type in _parent_type:
            OBJECT_TAG_REGISTER.append(
                (_tag_name, _p_type, _tag_type, _short_descr)
            )
    else:
        OBJECT_TAG_REGISTER.append(
            (_tag_name, _parent_type, _tag_type, _short_descr)
        )

OBJECT_TAG_TABLE = pd.DataFrame(OBJECT_TAG_REGISTER)
OBJECT_TAG_LIST = OBJECT_TAG_TABLE[0].unique().tolist()


def check_tag_is_valid(tag_name, tag_value):
    """Check validity of a tag value.

    Parameters
    ----------
    tag_name : str
        name of the tag, as it appears in an object's ``_tags`` dictionary
    tag_value : object
        value of the tag

    Raises
    ------
    KeyError
        if ``tag_name`` is not a registered tag
    ValueError
        if ``tag_value`` is not valid for the tag named ``tag_name``
    """
    if tag_name not in OBJECT_TAG_LIST:
        raise KeyError(f"{tag_name} is not a valid tag")

    tag_row = OBJECT_TAG_TABLE[OBJECT_TAG_TABLE[0] == tag_name]
    tag_type = tag_row.iloc[0, 2]

    if isinstance(tag_type, str):
        # no tag in the register declares a bare "list"; adding one means
        # adding its arm here and a case in test_check_tag_is_valid_rejects
        expected = {"bool": bool, "int": int, "str": str}[tag_type]
        # bool is a subclass of int, so an int tag must not accept True
        if tag_type == "int" and isinstance(tag_value, bool):
            raise ValueError(f"{tag_name} must be int, found bool")
        if not isinstance(tag_value, expected):
            raise ValueError(
                f"{tag_name} must be {tag_type}, found {type(tag_value).__name__}"
            )
        return

    kind, allowed = tag_type

    if kind == "str":
        if not isinstance(tag_value, str):
            raise ValueError(
                f"{tag_name} must be str, found {type(tag_value).__name__}"
            )
        if tag_value not in allowed:
            raise ValueError(f"{tag_name} must be one of {allowed}, found {tag_value}")
        return

    # the only remaining kind is "list"; the register is checked for shape by
    # test_every_tag_type_is_supported, so there is no unreachable arm here
    if allowed == "str":
        values = [tag_value] if isinstance(tag_value, str) else tag_value
        if not isinstance(values, list) or not all(
            isinstance(x, str) for x in values
        ):
            raise ValueError(f"{tag_name} must be a str or a list of str")
        return

    values = [tag_value] if isinstance(tag_value, str) else tag_value
    if not isinstance(values, list) or not set(values).issubset(allowed):
        raise ValueError(f"{tag_name} must be a subset of {allowed}")
