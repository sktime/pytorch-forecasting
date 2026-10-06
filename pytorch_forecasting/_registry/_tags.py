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
    Read by ``TestAllPtMetrics`` in ``metrics/tests/test_all_metrics.py``:
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


class info__metric_name(_BaseTag):
    """Human-readable metric name.

    Possible values
    ---------------
    Any str. By convention it matches the metric class name, e.g. "MASE"
    for ``MASE``.

    Default
    -------
    No default. Metric packages declare it.

    Effect
    ------
    Declarative only. No code in the repository reads this tag; unlike
    ``info:name``, which backs the ``name`` property of forecaster packages,
    this metric-side counterpart has no consumer. It is exposed through
    ``all_objects(return_tags=["info:metric_name"])``.
    """

    _tags = {
        "tag_name": "info:metric_name",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "human-readable metric name, matching the class name",
    }


class requires__data_type(_BaseTag):
    """Name of the test data fixture the metric is scored against.

    Possible values
    ---------------
    "point_forecast"
    "quantile_forecast"
    "classification_forecast"
    "beta_distribution_forecast"
    "log_normal_distribution_forecast"
    "mqf2_distribution_forecast"
    "multivariate_normal_distribution_forecast"
    "negative_binomial_distribution_forecast"
    "normal_distribution_forecast"
    "implicit_quantile_network_distribution_forecast"

    Each value names a fixture defined in ``metrics/tests/conftest.py``.

    Default
    -------
    No default. Metric packages must declare this tag.

    Effect
    ------
    Read by ``TestAllPtMetrics`` in ``metrics/tests/test_all_metrics.py``,
    where it selects the fixture that produces ``y_true`` and ``y_pred`` for
    this metric. A value with no matching fixture makes the metric's whole
    test class error out.
    """

    _tags = {
        "tag_name": "requires:data_type",
        "parent_type": "metric",
        "tag_type": (
            "str",
            [
                "point_forecast",
                "quantile_forecast",
                "classification_forecast",
                "beta_distribution_forecast",
                "log_normal_distribution_forecast",
                "mqf2_distribution_forecast",
                "multivariate_normal_distribution_forecast",
                "negative_binomial_distribution_forecast",
                "normal_distribution_forecast",
                "implicit_quantile_network_distribution_forecast",
            ],
        ),
        "short_descr": "name of the test data fixture the metric is scored against",
    }


class distribution_type(_BaseTag):
    """Distribution family a distributional metric assumes.

    Possible values
    ---------------
    "beta"
    "implicit_quantile_network"
    "log_normal"
    "mqf2"
    "multivariate_normal"
    "negative_binomial"
    "normal"

    Default
    -------
    No default. Only distributional metric packages declare it.

    Effect
    ------
    Read by ``TestAllPtMetrics`` in ``metrics/tests/test_all_metrics.py`` to
    build distribution parameters of the right shape before scoring.
    """

    _tags = {
        "tag_name": "distribution_type",
        "parent_type": "metric",
        "tag_type": (
            "str",
            [
                "beta",
                "implicit_quantile_network",
                "log_normal",
                "mqf2",
                "multivariate_normal",
                "negative_binomial",
                "normal",
            ],
        ),
        "short_descr": "distribution family a distributional metric assumes",
    }


class no_rescaling(_BaseTag):
    """Whether the metric is scored on unrescaled predictions.

    Possible values
    ---------------
    True   the metric is scored as is, without inverting the target
           transformation first
    False  the metric is scored after rescaling, the usual case

    Default
    -------
    No default; treated as ``False`` when absent.

    Effect
    ------
    Read by ``TestAllPtMetrics`` in ``metrics/tests/test_all_metrics.py``: when
    ``True``, the test harness skips the rescaling step before scoring.
    """

    _tags = {
        "tag_name": "no_rescaling",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether the metric is scored without rescaling",
    }


class capability__quantile_generation(_BaseTag):
    """Whether the metric can turn its prediction into quantiles.

    Possible values
    ---------------
    True   the metric exposes a quantile view of the prediction
    False  it does not

    Default
    -------
    No default; treated as ``False`` when absent.

    Effect
    ------
    Declarative only. No code in the repository branches on this tag. It is
    exposed for discovery through
    ``all_objects(filter_tags={"capability:quantile_generation": True})``.
    """

    _tags = {
        "tag_name": "capability:quantile_generation",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether the metric can generate quantiles",
    }


class shape__adds_quantile_dimension(_BaseTag):
    """Whether the metric's output carries an extra trailing quantile axis.

    Possible values
    ---------------
    True   the metric adds a trailing quantile dimension to its output
    False  the output keeps the shape of the input

    Default
    -------
    No default; treated as ``False`` when absent.

    Effect
    ------
    Declarative only. No code in the repository branches on this tag. It is
    exposed for discovery through
    ``all_objects(filter_tags={"shape:adds_quantile_dimension": True})``.
    """

    _tags = {
        "tag_name": "shape:adds_quantile_dimension",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether the metric output adds a quantile dimension",
    }


# --------------------------
# Scaler strategies
# --------------------------


class is_label_encoder(_BaseTag):
    """Whether the scaler strategy encodes labels rather than scaling values.

    Possible values
    ---------------
    True   the strategy maps categorical labels to integer codes
    False  the strategy applies a numeric transformation

    Default
    -------
    No default; ``ScalerAdapter`` falls back to ``False`` when the tag is
    absent and when no strategy is set.

    Effect
    ------
    Read by ``ScalerAdapter`` in ``adapters/scaler_adapters.py``. It drives
    ``ScalerAdapter.label_encoder_mask``, which tells the data pipeline which
    sub-normalizers are label encoders and must not be rescaled numerically.
    """

    _tags = {
        "tag_name": "is_label_encoder",
        "parent_type": "scaler_strategy",
        "tag_type": "bool",
        "short_descr": "whether the strategy encodes labels instead of scaling",
    }


class fit_per_sequence(_BaseTag):
    """Whether the scaler strategy is fitted separately for each sequence.

    Possible values
    ---------------
    True   statistics are computed per sequence, as for an encoder normalizer
    False  one set of statistics is fitted across the whole dataset

    Default
    -------
    No default; ``ScalerAdapter`` falls back to ``False`` when the tag is
    absent and when no strategy is set.

    Effect
    ------
    Read by ``ScalerAdapter`` in ``adapters/scaler_adapters.py``. For a
    multi-normalizer the adapter takes the logical or over its sub-adapters,
    so one per-sequence sub-normalizer makes the whole adapter per-sequence.
    """

    _tags = {
        "tag_name": "fit_per_sequence",
        "parent_type": "scaler_strategy",
        "tag_type": "bool",
        "short_descr": "whether the strategy is fitted separately per sequence",
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


class capability__multivariate(_BaseTag):
    """Whether the model supports multiple target variables.

    Possible values
    ---------------
    True   multivariate forecasting supported
    False  univariate target only

    Default
    -------
    No default. An undeclared tag reads back as ``None``, not ``False``.

    Effect
    ------
    Read by the ``model_overview`` sphinx extension
    (``docs/source/_ext/model_overview.py``), which fills the "Multiple
    targets" column of the generated model overview table. No runtime code
    branches on it.
    """

    _tags = {
        "tag_name": "capability:multivariate",
        "parent_type": "forecaster_pytorch",
        "tag_type": "bool",
        "short_descr": "whether the model supports multivariate targets",
    }


class capability__pred_int(_BaseTag):
    """Whether the model produces probabilistic prediction intervals.

    Possible values
    ---------------
    True   prediction intervals supported
    False  point forecasts only

    Default
    -------
    No default. An undeclared tag reads back as ``None``, not ``False``.

    Effect
    ------
    Read by the ``model_overview`` sphinx extension
    (``docs/source/_ext/model_overview.py``), which fills the "Prediction
    intervals" column of the generated model overview table. No runtime code
    branches on it.
    """

    _tags = {
        "tag_name": "capability:pred_int",
        "parent_type": "forecaster_pytorch",
        "tag_type": "bool",
        "short_descr": "whether the model supports prediction intervals",
    }


class capability__flexible_history_length(_BaseTag):
    """Whether the model accepts a variable-length encoder history.

    Possible values
    ---------------
    True   the model works with encoder windows of varying length
    False  the model requires a fixed encoder length

    Default
    -------
    No default. An undeclared tag reads back as ``None``, not ``False``.

    Effect
    ------
    Read by the ``model_overview`` sphinx extension
    (``docs/source/_ext/model_overview.py``), which fills the "Flexible
    History Length" column of the generated model overview table. No runtime
    code branches on it.
    """

    _tags = {
        "tag_name": "capability:flexible_history_length",
        "parent_type": "forecaster_pytorch",
        "tag_type": "bool",
        "short_descr": "whether the model accepts variable-length history",
    }


class capability__cold_start(_BaseTag):
    """Whether the model can forecast with little or no history.

    Possible values
    ---------------
    True   the model produces forecasts for series it has not seen a long
           history of
    False  the model needs a full encoder history

    Default
    -------
    No default. An undeclared tag reads back as ``None``, not ``False``.

    Effect
    ------
    Read by the ``model_overview`` sphinx extension
    (``docs/source/_ext/model_overview.py``), which fills the "Cold Start"
    column of the generated model overview table. No runtime code branches
    on it.
    """

    _tags = {
        "tag_name": "capability:cold_start",
        "parent_type": "forecaster_pytorch",
        "tag_type": "bool",
        "short_descr": "whether the model can forecast with little history",
    }


# --------------------------
# Forecaster information
# --------------------------


class info__name(_BaseTag):
    """Human-readable model name.

    Possible values
    ---------------
    Any str. By convention it matches the model class name, e.g. "NBeats"
    for ``NBeats``, so that the documentation and the class agree.

    Default
    -------
    No default. Forecaster packages must declare this tag.

    Effect
    ------
    Backs the ``name`` property on ``_BasePtForecaster_Common``
    (``models/base/_base_object.py``). The ``model_overview`` sphinx
    extension also uses it to skip base and internal classes: an object
    without this tag is left out of the model overview table.
    """

    _tags = {
        "tag_name": "info:name",
        "parent_type": "forecaster_pytorch",
        "tag_type": "str",
        "short_descr": "human-readable model name, matching the class name",
    }


class info__compute(_BaseTag):
    """Approximate compute cost of training the model.

    Possible values
    ---------------
    1  lightweight, e.g. a plain MLP
    3  medium
    5  very heavy

    Values between those anchors are allowed. Only 1 to 4 occur today.

    Default
    -------
    No default. Forecaster packages must declare this tag.

    Effect
    ------
    Read by the ``model_overview`` sphinx extension
    (``docs/source/_ext/model_overview.py``), which fills the "Compute (1-5)"
    column of the generated model overview table. No runtime code branches
    on it, and nothing enforces the 1 to 5 range.
    """

    _tags = {
        "tag_name": "info:compute",
        "parent_type": "forecaster_pytorch",
        "tag_type": "int",
        "short_descr": "approximate compute cost, 1 (light) to 5 (very heavy)",
    }


class info__pred_type(_BaseTag):
    """Kinds of prediction the model produces.

    Possible values
    ---------------
    "point"     deterministic point forecasts
    "quantile"  probabilistic quantile forecasts
    "distr"     a full predictive distribution, e.g. DeepAR

    The value is a list; a model may declare more than one.

    Default
    -------
    No default. An undeclared tag reads back as ``None``; consumers fall back
    to an empty list. The v2 packages currently omit it, although the v2
    extension template asks for it.

    Effect
    ------
    Read by ``EstimatorFixtureGenerator._get_compatible_losses_for_model`` in
    ``tests/test_all_estimators.py``, which passes it to
    ``get_compatible_losses`` to decide which loss functions the model is
    tested against. Also read by the ``model_overview`` sphinx extension to
    fill the "Probabilistic" column.
    """

    _tags = {
        "tag_name": "info:pred_type",
        "parent_type": "forecaster_pytorch",
        "tag_type": ("list", ["point", "quantile", "distr"]),
        "short_descr": "kinds of prediction the model produces",
    }


class info__y_type(_BaseTag):
    """Kinds of target the model supports.

    Possible values
    ---------------
    "numeric"   continuous or numeric target variables
    "category"  categorical target variables, e.g. for classification losses

    The value is a list; a model may declare more than one.

    Default
    -------
    No default. An undeclared tag reads back as ``None``; consumers fall back
    to an empty list.

    Effect
    ------
    Read by ``EstimatorFixtureGenerator._get_compatible_losses_for_model`` in
    ``tests/test_all_estimators.py``, which passes it to
    ``get_compatible_losses`` to decide which loss functions the model is
    tested against. Also read by the ``model_overview`` sphinx extension to
    fill the "Regression" and "Classification" columns.
    """

    _tags = {
        "tag_name": "info:y_type",
        "parent_type": "forecaster_pytorch",
        "tag_type": ("list", ["numeric", "category"]),
        "short_descr": "kinds of target the model supports",
    }


# --------------------------
# Packaging and testing
# --------------------------


class authors(_BaseTag):
    """GitHub handles of the contributors of the object.

    Possible values
    ---------------
    A list of str, each a GitHub handle. Handles of authors of code ported
    from another package are included, so the list is attribution, not a
    maintainer roster.

    Default
    -------
    No default.

    Effect
    ------
    Returned by ``all_objects(return_tags=["authors"])`` and shown in the
    model overview table built by the ``model_overview`` sphinx extension.
    No runtime code branches on it.
    """

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "GitHub handles of the contributors of the object",
    }


class python_dependencies(_BaseTag):
    """External packages the object needs, beyond the core dependencies.

    Possible values
    ---------------
    A list of str, each a PEP 440 requirement string, e.g. ``["cpflows"]``.
    An empty list, or an absent tag, means the object runs on the core
    dependencies alone.

    Default
    -------
    No default; treated as no extra dependencies when absent.

    Effect
    ------
    Consumed by the soft-dependency machinery in ``utils/_dependencies``,
    which skips tests and raises an actionable import error when a declared
    package is missing.
    """

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "external packages required by the object",
    }


class tests__skip_by_name(_BaseTag):
    """Test cases to skip for this object, by full test name.

    Possible values
    ---------------
    A list of str, each the name of a test as pytest reports it, including
    the parametrisation, e.g.
    ``"test_integration[NHiTS-base_params-0-NormalDistributionLoss]"``.
    A bare test name such as ``"test_integration"`` skips every
    parametrisation of that test.

    Default
    -------
    No default; treated as an empty list when absent.

    Effect
    ------
    Read by ``EstimatorFixtureGenerator.is_excluded`` in
    ``tests/test_all_estimators.py``, which drops the named cases from the
    generated test matrix for this object. For a model class rather than a
    package the tag is read from its ``pkg`` attribute.
    """

    _tags = {
        "tag_name": "tests:skip_by_name",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "test cases to skip for this object, by full test name",
    }


def _build_tag_register():
    """Collect every tag class in this module into register rows.

    Returns
    -------
    list of tuple
        one ``(tag_name, parent_type, tag_type, short_descr)`` per tag and
        parent type; a tag declaring several parent types yields one row each
    """
    register = []
    for _, cl in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if cl is _BaseTag or not issubclass(cl, _BaseTag):
            continue

        cl_tags = cl.get_class_tags()
        tag_name = cl_tags["tag_name"]
        parent_type = cl_tags["parent_type"]
        tag_type = cl_tags["tag_type"]
        short_descr = cl_tags["short_descr"]

        if isinstance(parent_type, list):
            for p_type in parent_type:
                register.append((tag_name, p_type, tag_type, short_descr))
        else:
            register.append((tag_name, parent_type, tag_type, short_descr))
    return register


OBJECT_TAG_REGISTER = _build_tag_register()
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
        if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
            raise ValueError(f"{tag_name} must be a str or a list of str")
        return

    values = [tag_value] if isinstance(tag_value, str) else tag_value
    if not isinstance(values, list) or not set(values).issubset(allowed):
        raise ValueError(f"{tag_name} must be a subset of {allowed}")
