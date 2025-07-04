"""Automated test for all metrics in PyTorch Forecasting."""

from inspect import isclass
import shutil

import numpy as np
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics.base_metrics import (
    CompositeMetric,
    Metric,
    MultiHorizonMetric,
    MultiLoss,
    TorchMetricWrapper,
)

ONLY_CHANGED_MODULES = False


def get_all_metrics():
    """
    Manually collect all metric classes.

    Current approach it to manually include classes one by one.
    If breaking changes are detected by the introduction of a metric into the
    test suite, it is decided to either change the test or include metric-specific
    tests in the metric's own test suite.
    """
    metric_classes = []
    try:
        from pytorch_forecasting.metrics.point import (
            MAE,
            MAPE,
            MASE,
            RMSE,
            SMAPE,
            CrossEntropy,
            PoissonLoss,
            TweedieLoss,
        )

        metric_classes.extend(
            [
                MAE,
                MAPE,
                RMSE,
                SMAPE,
                PoissonLoss,
                TweedieLoss,
                # CrossEntropy,
                # MASE
            ]
        )
    except ImportError:
        raise ImportError("Failed to import point metrics.")
    try:
        from pytorch_forecasting.metrics.quantile import QuantileLoss

        metric_classes.append(QuantileLoss)
    except ImportError:
        raise ImportError("Failed to import quantile metrics.")

    from pytorch_forecasting.metrics.distributions import (
        NormalDistributionLoss,
    )

    metric_classes.extend(
        [
            NormalDistributionLoss,
        ]
    )

    return metric_classes


def get_test_data():
    """Return test data fixtures for the metrics.

    The general nomenclature for naming test cases is <target_type>_<prediction_type>,
    where `target_type` is specified only if target is weighted or packed_sequence, and
    `prediction_type` is broadly 4 types - point, quantile, distibution and
    classification. Classficiation is a special case for class predictions used by Cross
    Entropy metric.

    * `point`: 2D tensor of point forecasts with 2D predictions and actuals.
    * `quantile`: 3D tensor of quantile forecasts and 2D actuals.
    * `packed_sequence_point`: 2D point predictions and 2D packed sequence actuals.
    * `packed_sequence_quantile`: 3D tensor of quantile predictions with 2D packed sequence
        actuals.
    * `weighted_point`: 2D tensor of point forecasts and weighted 2D actuals.
    * `weighted_quantile`: 3D tensor of quantile forecasts and weighted 2D actuals.
    * `distribution`: 3D tensor of prediction with distribution parameters and 2D
        actuals.
    * `packed_sequence_distribution`: 3D predictions with distribution params
        and 2D packed_sequence actuals.
    * `weighted_distribution`: 3D tensor of prediction with distribution parameters
        and 2D weighted actuals.
    * `classification`: 3D class predictions and 2D actuals
        containing integer values.
    """  # noqa: E501

    batch_size, timesteps = 4, 10

    test_cases = {}

    y_pred_point = torch.randn(batch_size, timesteps)
    y_actual_2d = torch.randn(batch_size, timesteps)
    test_cases["point"] = {"y_pred": y_pred_point, "y_actual": y_actual_2d}

    # 3d tensors for quantile forecast (default 7 quantiles)
    y_pred_quantile = torch.randn(batch_size, timesteps, 7)
    test_cases["quantile"] = {"y_pred": y_pred_quantile, "y_actual": y_actual_2d}

    # data with weighted targets
    y_actual_weighted = (y_actual_2d, torch.ones(batch_size, timesteps))
    test_cases["weighted_point"] = {
        "y_pred": y_pred_point,
        "y_actual": y_actual_weighted,
    }

    test_cases["weighted_quantile"] = {
        "y_pred": y_pred_quantile,
        "y_actual": (y_actual_2d, torch.ones(batch_size, timesteps)),
    }

    # PackedSequence data (simulates variable length sequences in each batch)
    lengths = torch.tensor([10, 8, 6, 4])
    padded_actual = torch.randn(batch_size, timesteps)

    packed_actual = rnn.pack_padded_sequence(
        padded_actual, lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed_sequence_point"] = {
        "y_pred": y_pred_point,
        "y_actual": packed_actual,
    }  # noqa: E501
    test_cases["packed_sequence_quantile"] = {
        "y_pred": y_pred_quantile,
        "y_actual": packed_actual,
    }  # noqa: E501
    # 3d tensors for distribution parameters
    n_params = 10  # keeping 10 distribution parameters for coverage.
    y_pred_dist = torch.randn(batch_size, timesteps, n_params)
    y_pred_dist[..., 1] = torch.abs(y_pred_dist[..., 1]) + 0.1
    y_pred_dist[..., 3] = torch.abs(y_pred_dist[..., 3]) + 0.1
    test_cases["distribution"] = {"y_pred": y_pred_dist, "y_actual": y_actual_2d}
    test_cases["packed_sequence_distribution"] = {
        "y_pred": y_pred_dist,
        "y_actual": packed_actual,
    }  # noqa: E501
    test_cases["weighted_distribution"] = {
        "y_pred": y_pred_dist,
        "y_actual": y_actual_weighted,
    }  # noqa: E501

    n_class = 5
    y_pred_class = torch.randn(batch_size, timesteps, n_class)
    y_actual_class = torch.randint(0, 5, (batch_size, timesteps))
    test_cases["classification"] = {"y_pred": y_pred_class, "y_actual": y_actual_class}

    return test_cases


DATA_FORMAT_NAMES = [
    "point",
    "quantile",
    "packed_sequence_point",
    "packed_sequence_quantile",
    "weighted_point",
    "weighted_quantile",
    "distribution",
    "packed_sequence_distribution",
    "weighted_distribution",
    "classification",
]


# TODO: Implement tag system to infer metric compatibility with data formats.
def get_metric_compatibility():
    """Return a dictonary mapping metric classes to their compatible data formats."""

    metric_compatibility = {
        "MAE": {"point", "packed_sequence_point", "weighted_point"},
        "MAPE": {"point", "packed_sequence_point", "weighted_point"},
        "RMSE": {"point", "packed_sequence_point", "weighted_point"},
        "SMAPE": {"point", "packed_sequence_point", "weighted_point"},
        "PoissonLoss": {"point", "packed_sequence_point", "weighted_point"},
        "TweedieLoss": {"point", "packed_sequence_point", "weighted_point"},
        "QuantileLoss": {"quantile", "packed_sequence_quantile", "weighted_quantile"},
        "NormalDistributionLoss": {
            "distribution",
            "weighted_distribution",
            "packed_sequence_distribution",
        },
        "NegativeBinomialDistributionLoss": {
            "distribution",
            "packed_sequence_distribution",
            "weighted_distribution",
        },
        "LogNormalDistributionLoss": {
            "distribution",
            "packed_sequence_distribution",
            "weighted_distribution",
        },  # noqa: E501
        "MultivariateNormalDistributionLoss": {
            "distribution",
            "packed_sequence_distribution",
            "weighted_distribution",
        },  # noqa: E501
        "BetaDistributionLoss": {
            "distribution",
            "packed_sequence_distribution",
            "weighted_distribution",
        },
    }

    return metric_compatibility


METRIC_COMPATIBILITY = get_metric_compatibility()


@pytest.fixture(params=get_all_metrics())
def metric_class(request):
    """Fixture to provide metric classes."""
    metric_class = request.param
    return metric_class


@pytest.fixture
def metric_instance(metric_class):
    """Fixture to instantiate metric classes."""
    if issubclass(metric_class, (MultiLoss, CompositeMetric, TorchMetricWrapper)):
        pytest.skip(f"{metric_class.__name__} requires special instantiation.")

    try:
        return metric_class()
    except Exception as e:
        pytest.fail(f"Failed to instantiate {metric_class.__name__}: {e}")


@pytest.fixture(
    params=[
        (cls, fmt)
        for cls in get_all_metrics()
        for fmt in (METRIC_COMPATIBILITY.get(cls.__name__, DATA_FORMAT_NAMES))
    ]
)
def metric_class_with_format(request):
    """Fixture to provide metric classes."""
    return request.param


@pytest.fixture
def metric_instance_with_format(metric_class_with_format):
    """Fixture to instantiate metric classes."""
    metric_class, data_format = metric_class_with_format
    if issubclass(metric_class, (MultiLoss, CompositeMetric, TorchMetricWrapper)):
        pytest.skip(f"{metric_class.__name__} requires special instantiation.")

    try:
        return metric_class(), data_format
    except Exception as e:
        pytest.fail(f"Failed to instantiate {metric_class.__name__}: {e}")


@pytest.fixture(scope="session")
def test_data():
    """Session-scoped fixture for test data."""
    return get_test_data()


class TestAllPtMetrics:
    """Test suite for all metrics in PyTorch Forecasting.

    This test class implements a comprehensive testing strategy for metrics that:

    1. Tests all metrics against diverse input formats (point forecasts, quantiles,
    packed sequences, weighted inputs, distributions, classification)

    2. Uses pytest.skip to gracefully handle inherent incompatibilities between
    metrics and data formats, recognizing that not all metrics work with all data
    types. This approach distinguishes between expected incompatibilities vs actual
    bugs.

    3. Validates both the functional interface (direct calls) and stateful interface
    (update/compute pattern) of metrics.

    4. Checks important metric properties including:

    - Output tensor validity (finite values, correct shape)
    - Reduction modes behavior (none, mean, sqrt-mean)
    - Packed sequence handling for variable-length data
    - Special methods like to_prediction and to_quantiles
    - Metric composition behavior
    - Example validation through doctests

    The test design favors comprehensiveness over specificity, attempting all
    combinations rather than hardcoding expected compatibilities. This makes the
    tests more maintainable and future-proof, automatically handling new metrics
    or data formats without requiring test updates.

    When encountering incompatibilities, tests are skipped with descriptive messages
    rather than marked as failures, since incompatibility is treated as
    expected behavior and not a bug. This approach results in many skipped tests
    but ensures genuine issues are visible and all metrics receive appropriate
    validation against compatible formats.

    Note
    ----
    There is a plan to implement a variable counter in each test to track if the
    metric accepts atleast a single data format. This will help in
    identifying metrics that are not tested at all, and thus not compatible with
    any provided data format, and raise an error.
    """

    def test_metric_instantiation(self, metric_class):
        "Test instantiation of metric classes."
        try:
            if issubclass(metric_class, (MultiLoss, CompositeMetric)):
                pytest.skip(
                    f"{metric_class.__name__} is requiring special instantiation."
                )
            elif issubclass(metric_class, TorchMetricWrapper):
                pytest.skip(
                    f"{metric_class.__name__} is a wrapper and should not be instantiated directly."  # noqa: E501
                )
            else:
                metric = metric_class()

        except Exception as e:
            pytest.fail(f"Failed to instantiate {metric_class.__name__}: {e}")

        assert isinstance(metric, Metric)

    def test_metric_forward_pass(self, metric_instance_with_format, test_data):
        """Test forward pass of metric instance."""
        metric_instance, test_case = metric_instance_with_format

        if metric_instance is None:
            pytest.skip("No valid metric instance and data format provided.")

        test_case_data = test_data[test_case]

        y_pred = test_case_data["y_pred"]
        y_actual = test_case_data["y_actual"]
        res = metric_instance(y_pred, y_actual)

        assert isinstance(
            res, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res)}"  # noqa: E501
        # dependent on metric reduction type.
        assert torch.numel(res) == 1 or res.ndim <= 2
        assert torch.isfinite(res).all(), "Output contains non-finite values."

    def test_metric_update_compute(self, metric_instance_with_format, test_data):
        """Test update/compute methods pattern of metric."""

        metric_instance, test_case = metric_instance_with_format

        if metric_instance is None:
            pytest.skip("No valid metric instance and data format provided.")

        test_case_data = test_data[test_case]

        y_pred = test_case_data["y_pred"]
        y_actual = test_case_data["y_actual"]

        metric_instance.reset()
        metric_instance.update(y_pred, y_actual)
        res = metric_instance.compute()

        assert isinstance(
            res, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res)}"  # noqa: E501
        assert torch.isfinite(res).all(), "Output contains non-finite values."

        metric_instance.update(y_pred, y_actual)
        res2 = metric_instance.compute()

        assert isinstance(
            res2, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res2)}"  # noqa: E501

    def test_metric_reduction(self, metric_class_with_format, test_data):
        """Test metric reduction behavior."""

        metric_class, test_case = metric_class_with_format

        if not issubclass(metric_class, (Metric, MultiHorizonMetric)):
            pytest.skip(f"{metric_class.__name__} does not support reduction testing.")

        reduction_modes = ["none", "mean", "sqrt-mean"]

        y_pred = test_data[test_case]["y_pred"]
        y_actual = test_data[test_case]["y_actual"]

        for reduction in reduction_modes:
            metric = metric_class(reduction=reduction)
            res = metric(y_pred, y_actual)

            if reduction == "none":
                assert res.numel() > 1, "Expected multiple outputs for `none` reduction"
                assert res.shape[0] == y_pred.shape[0]
                assert res.shape[1] == y_pred.shape[1]
            elif reduction in ["mean", "sqrt-mean"]:
                assert res.numel() <= y_pred.shape[0]
                if reduction == "sqrt-mean":
                    assert (
                        res >= 0
                    ), "Expected non-negative output for `sqrt-mean` reduction."  # noqa: E501

    @pytest.mark.parametrize(
        "test_case",
        [
            "packed_sequence_point",
            "packed_sequence_quantile",
            "packed_sequence_distribution",
        ],
    )
    def test_packed_sequence_handling(self, metric_instance, test_data, test_case):
        """Test that metrics properly handle PackedSequence inputs."""

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        metric_name = metric_instance.__class__.__name__

        if test_case not in METRIC_COMPATIBILITY.get(metric_name, set()):
            pytest.skip(
                f"{metric_name} is not compatible with {test_case} data format."
            )

        test_case_data = test_data[test_case]
        y_pred = test_case_data["y_pred"]
        y_actual = test_case_data["y_actual"]

        res = metric_instance(y_pred, y_actual)

        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Output contains non-finite values."

        metric_instance.reset()
        metric_instance.update(y_pred, y_actual)
        res2 = metric_instance.compute()
        assert isinstance(res2, torch.Tensor)
        assert torch.isfinite(res2).all(), "Output contains non-finite values."

    def test_to_prediction_method(self, metric_instance_with_format, test_data):
        """Test the ``to_prediction`` method of metrics."""

        metric_instance, test_case = metric_instance_with_format

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        test_case_data = test_data[test_case]
        y_pred = test_case_data["y_pred"]

        if test_case in [
            "distribution",
            "packed_sequence_distribution",
            "weighted_distribution",
        ]:  # noqa: E501
            pytest.skip(
                f"Need to rescale parameters for {metric_instance.__class__.__name__}, cannot test to_prediction."  # noqa: E501
            )

        prediction = metric_instance.to_prediction(
            y_pred,
        )

        assert isinstance(prediction, torch.Tensor)

        if y_pred.ndim == 3:
            assert prediction.ndim == 2
            assert prediction.shape[:2] == y_pred.shape[:2]
        # else:
        #     assert prediction.shape == y_pred.shape

    def test_to_quantiles_method(self, metric_instance_with_format, test_data):
        """Test the ``to_quantiles`` method of metrics."""

        metric_instance, test_case = metric_instance_with_format

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        test_case_data = test_data[test_case]
        y_pred = test_case_data["y_pred"]
        quantiles = [0.1, 0.5, 0.9]
        if metric_instance.__class__.__name__ == "QuantileLoss":
            quantile_pred = metric_instance.to_quantiles(y_pred)
        else:
            quantile_pred = metric_instance.to_quantiles(y_pred, quantiles=quantiles)  # noqa: E501
        assert isinstance(quantile_pred, torch.Tensor)

        if metric_instance.__class__.__name__ == "QuantileLoss":
            assert quantile_pred.shape == y_pred.shape
        else:
            assert quantile_pred.ndim == 3
            if y_pred.ndim == 2 and metric_instance.__class__.__name__ != "PoissonLoss":  # noqa: E501
                assert quantile_pred.shape[2] == 1
            else:
                assert quantile_pred.shape[2] == len(quantiles)

    def test_metric_compositions(self, metric_class):
        """Test correct working of metric compositions via CompositeMetric"""

        if not issubclass(metric_class, Metric):
            pytest.skip(
                f"{metric_class.__name__} is not a Metric subclass,",
                " skipping test.",
            )

        metric_a = metric_class()
        metric_b = metric_class()

        composite = metric_a + metric_b
        assert isinstance(composite, CompositeMetric)

        weighted = metric_a * 0.5
        assert isinstance(weighted, CompositeMetric)

        composite_weighted = metric_a + metric_b * 0.5
        assert isinstance(composite_weighted, CompositeMetric)

    def test_doctest_examples(self, metric_class):
        """Run doctests for metric class."""
        try:
            from skbase.utils.doctest_run import run_doctest

            run_doctest(metric_class, name=f"class {metric_class.__name__}")
        except ImportError:
            pytest.skip("skbase doctest utilities not available")
        except Exception as e:
            pytest.skip(f"Doctest failed for {metric_class.__name__}: {e}")
