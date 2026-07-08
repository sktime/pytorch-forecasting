"""Register of estimator and object tags.

Note for extenders: new tags should be entered in OBJECT_TAG_REGISTER.
No other place is necessary to add new tags.

This module exports the following:

---
OBJECT_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - name of the object_type this tag applies to,
                 e.g., "forecaster_pytorch_v1", "forecaster_pytorch_v2", "metric",
                 or "object" if the tag applies to all objects
    2 : string - expected type of the tag value
        should be one of:
            "bool" - valid values are True/False
            "int" - valid values are all integers
            "str" - valid values are all strings
            "list" - valid values are all lists of arbitrary elements
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
            ("list", "str") - any individual string or list of strings is valid
        validity can be checked by check_tag_is_valid (see below)
    3 : string - plain English description of the tag

---

OBJECT_TAG_TABLE - pd.DataFrame
    OBJECT_TAG_REGISTER in table form, as pd.DataFrame
        rows of OBJECT_TAG_TABLE correspond to elements in OBJECT_TAG_REGISTER

OBJECT_TAG_LIST - list of string
    elements are 0-th entries of OBJECT_TAG_REGISTER, in same order

---

check_tag_is_valid(tag_name, tag_value) - checks whether tag_value is valid for tag_name
"""

import inspect
import sys

import pandas as pd

from pytorch_forecasting.base._base_object import _BaseObject


class _BaseTag(_BaseObject):
    """Base class for all tags in pytorch-forecasting.

    This follows the class-based tag registry pattern used in sktime and skpro,
    for better extensibility and single-source-of-truth tag documentation.
    """

    _tags = {
        "object_type": "tag",
        "tag_name": "",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "",
    }


_FORECASTER_TYPES = ["forecaster_pytorch_v1", "forecaster_pytorch_v2"]


class object_type(_BaseTag):
    """Type of object, e.g., 'forecaster_pytorch_v1', 'metric'."""

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": (
            "str",
            ["forecaster_pytorch_v1", "forecaster_pytorch_v2", "metric"],
        ),
        "short_descr": "type of object, e.g., forecaster_pytorch_v1, metric",
    }


class authors(_BaseTag):
    """List of GitHub handles of the object's authors."""

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "list of GitHub handles of the object's authors",
    }


class python_dependencies(_BaseTag):
    """List of external Python packages required by the object."""

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "list of external python packages required by the object",
    }


class tests_skip_by_name(_BaseTag):
    """Names of tests to skip in CI for this object."""

    _tags = {
        "tag_name": "tests:skip_by_name",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "names of tests to skip in CI for this object",
    }


class info_name(_BaseTag):
    """Human-readable model name."""

    _tags = {
        "tag_name": "info:name",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "str",
        "short_descr": "human-readable model name, matching the class name",
    }


class info_compute(_BaseTag):
    """Compute intensity of the model."""

    _tags = {
        "tag_name": "info:compute",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "int",
        "short_descr": "compute intensity of the model, from 1 (light) to 5 (heavy)",
    }


class info_pred_type(_BaseTag):
    """Prediction types the model produces."""

    _tags = {
        "tag_name": "info:pred_type",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": ("list", ["point", "quantile", "distr"]),
        "short_descr": (
            "prediction types the model produces, e.g. point, quantile, distr"
        ),
    }


class info_y_type(_BaseTag):
    """Target data type(s) the model supports."""

    _tags = {
        "tag_name": "info:y_type",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "list",
        "short_descr": "target data type(s) the model supports, e.g. numeric, category",
    }


class capability_cold_start(_BaseTag):
    """Whether the model can forecast with little or no history."""

    _tags = {
        "tag_name": "capability:cold_start",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "bool",
        "short_descr": "whether the model can forecast with little or no history",
    }


class capability_exogenous(_BaseTag):
    """Whether the model supports exogenous variables."""

    _tags = {
        "tag_name": "capability:exogenous",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "bool",
        "short_descr": "whether the model supports exogenous variables",
    }


class capability_flexible_history_length(_BaseTag):
    """Whether the model works with variable-length input history."""

    _tags = {
        "tag_name": "capability:flexible_history_length",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "bool",
        "short_descr": "whether the model works with variable-length input history",
    }


class capability_multivariate(_BaseTag):
    """Whether the model supports multivariate targets."""

    _tags = {
        "tag_name": "capability:multivariate",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "bool",
        "short_descr": "whether the model supports multivariate targets",
    }


class capability_pred_int(_BaseTag):
    """Whether the model produces prediction intervals."""

    _tags = {
        "tag_name": "capability:pred_int",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "bool",
        "short_descr": "whether the model produces prediction intervals",
    }


class capability_quantile_generation(_BaseTag):
    """Whether the model supports quantile output generation."""

    _tags = {
        "tag_name": "capability:quantile_generation",
        "parent_type": _FORECASTER_TYPES,
        "tag_type": "bool",
        "short_descr": "whether the model supports quantile output generation",
    }


class info_metric_name(_BaseTag):
    """Human-readable metric name."""

    _tags = {
        "tag_name": "info:metric_name",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "human-readable metric name",
    }


class metric_type(_BaseTag):
    """Type of metric, e.g. point, point_classification, distribution."""

    _tags = {
        "tag_name": "metric_type",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": (
            "type of metric, e.g. point, point_classification, distribution, quantile"
        ),
    }


class distribution_type(_BaseTag):
    """Distribution family used by a distributional loss."""

    _tags = {
        "tag_name": "distribution_type",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "distribution family used by a distributional loss, e.g. normal",
    }


class requires_data_type(_BaseTag):
    """Expected input data format for the metric."""

    _tags = {
        "tag_name": "requires:data_type",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": (
            "expected input data format for the metric, "
            "e.g. point_forecast, classification_forecast"
        ),
    }


class no_rescaling(_BaseTag):
    """Whether the metric is invariant to rescaling."""

    _tags = {
        "tag_name": "no_rescaling",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether the metric is invariant to rescaling of the target",
    }


class shape_adds_quantile_dimension(_BaseTag):
    """Whether the metric output adds an extra quantile axis."""

    _tags = {
        "tag_name": "shape:adds_quantile_dimension",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether the metric output adds an extra quantile axis",
    }


OBJECT_TAG_REGISTER = []
tag_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for _, cl in tag_classes:
    if cl.__name__ == "_BaseTag" or not issubclass(cl, _BaseTag):
        continue

    cl_tags = cl.get_class_tags()
    tag_name = cl_tags.get("tag_name", "unknown_tag")
    parent_type = cl_tags.get("parent_type", "object")
    tag_type = cl_tags.get("tag_type", "str")
    short_descr = cl_tags.get("short_descr", "")

    if isinstance(parent_type, list):
        for p_type in parent_type:
            OBJECT_TAG_REGISTER.append((tag_name, p_type, tag_type, short_descr))
    else:
        OBJECT_TAG_REGISTER.append((tag_name, parent_type, tag_type, short_descr))

OBJECT_TAG_TABLE = pd.DataFrame(OBJECT_TAG_REGISTER)
OBJECT_TAG_LIST = OBJECT_TAG_TABLE[0].unique().tolist()


def check_tag_is_valid(tag_name, tag_value):
    """Check validity of a tag value.

    Parameters
    ----------
    tag_name : string, name of the tag
    tag_value : object, value of the tag

    Raises
    ------
    KeyError - if tag_name is not a valid tag in OBJECT_TAG_LIST
    ValueError - if the tag_value is not valid for the tag with name tag_name
    """
    if tag_name not in OBJECT_TAG_LIST:
        raise KeyError(f"{tag_name} is not a valid tag")

    tag_row = OBJECT_TAG_TABLE[OBJECT_TAG_TABLE[0] == tag_name]
    tag_type = tag_row.iloc[0, 2]

    if isinstance(tag_type, str):
        if tag_type == "bool" and not isinstance(tag_value, bool):
            raise ValueError(f"{tag_name} must be bool, found {type(tag_value)}")
        if tag_type == "int" and not isinstance(tag_value, int):
            raise ValueError(f"{tag_name} must be int, found {type(tag_value)}")
        if tag_type == "str" and not isinstance(tag_value, str):
            raise ValueError(f"{tag_name} must be str, found {type(tag_value)}")
        if tag_type == "list" and not isinstance(tag_value, list):
            raise ValueError(f"{tag_name} must be list, found {type(tag_value)}")

    elif isinstance(tag_type, tuple):
        if tag_type[0] == "str":
            if tag_value not in tag_type[1]:
                raise ValueError(
                    f"{tag_name} must be one of {tag_type[1]}, found {tag_value}"
                )

        elif tag_type[0] == "list" and tag_type[1] == "str":
            if not isinstance(tag_value, (str, list)):
                raise ValueError(
                    f"{tag_name} must be str or list of str, found {type(tag_value)}"
                )
            if isinstance(tag_value, list) and not all(
                isinstance(x, str) for x in tag_value
            ):
                raise ValueError(f"{tag_name} must be a list of strings.")

        elif tag_type[0] == "list" and isinstance(tag_type[1], list):
            if not isinstance(tag_value, list) or not set(tag_value).issubset(
                tag_type[1]
            ):
                raise ValueError(f"{tag_name} must be a subset of {tag_type[1]}")
