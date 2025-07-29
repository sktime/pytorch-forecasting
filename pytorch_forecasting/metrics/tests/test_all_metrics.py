"""Automated test for all metrics in PyTorch Forecasting."""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies
import torch

from pytorch_forecasting.metrics.tests._config import (
    EXCLUDE_METRICS,
    EXCLUDED_TESTS,
)
from pytorch_forecasting.tests._base._fixture_generator import (
    BaseFixtureGenerator,
)


class MetricPackageConfig:
    """Configuration for the metric package tests."""

    # This class can be extended to include specific configurations for the metric
    # package if needed in the future.

    package_name = "pytorch_forecasting.metrics"

    exclude_objects = EXCLUDE_METRICS
    excluded_tests = EXCLUDED_TESTS


class MetricFixtureGenerator(BaseFixtureGenerator):
    """
    Fixture generator for testing metrics in PyTorch Forecasting.

    Inherits from BaseFixtureGenerator to provide a framework for
    generating fixtures for metric classes and instances, to be tested under
    metric-specific scenarios.

    Fixtures parametrized
    ---------------------
    object_class: metric inheriting from BaseObject
        ranges over metric classes not excluded by EXCLUDE_METRICS, EXCLUDED_TESTS
    """

    fixture_sequence = ["object_pkg", "object_class", "object_instance"]

    @staticmethod
    def _check_required_dependencies(object_pkg):
        """
        Skip tests if the required dependencies for the metric are not installed
        in your environment.
        """

        required_deps = object_pkg.get_class_tag("python_dependencies")
        if required_deps:
            return _check_soft_dependencies(required_deps, severity="none")
        return True

    def _generate_object_instance(self, test_name, **kwargs):
        """Generate instance of the metric class for testing parametrized
        with scenarios in get_metric_test_params().
        """

        if "object_pkg" in kwargs:
            obj_meta = kwargs["object_pkg"]
        else:
            return []

        if not self._check_required_dependencies(obj_meta):
            return [], []

        metric_instances = []
        all_metric_test_params = obj_meta.get_metric_test_params()
        metric_class = obj_meta.get_cls()

        if not all_metric_test_params:
            metric_instances = [metric_class()]
            metric_instance_names = ["default"]
        else:
            rg = range(len(all_metric_test_params))
            metric_instances = [
                metric_class(**params) for params in all_metric_test_params
            ]

            metric_instance_names = [str(i) for i in rg]

        return metric_instances, metric_instance_names


class TestAllPtMetrics(MetricPackageConfig, MetricFixtureGenerator):
    """Test suite for all metrics in PyTorch Forecasting."""

    object_type_filter = "metric"

    def _setup_metric_test_scenario(
        self,
        object_pkg,
        object_instance,
        target_type,
        request,
    ):
        """Prepare test inputs for the given metric.

        Parameters
        ----------
        object_pkg: SkbaseBaseObject
            The package object containing the metric.
        object_instance: Metric
            An instance of the metric class to be tested.
        target_type: str
            The type of target data (e.g., "standard", "packed", "weighted").
        request: pytest.FixtureRequest
            The pytest request object to access fixtures.
        Returns
        -------
        tuple or None
            A tuple containing the prepared metric, y_pred and y, or None
            if the target type is not supported.
        """

        prepare_data_fixture_name = object_pkg.requires_data_type()
        data = request.getfixturevalue(prepare_data_fixture_name)

        if target_type not in data["test_cases"]:
            return None

        test_case = data["test_cases"][target_type]
        y_pred, y = object_pkg.prepare_test_inputs(test_case)

        metric = object_instance

        torch_encoder = object_pkg.get_encoder()
        if not object_pkg.get_class_tag("no_rescaling"):
            y_pred = metric.rescale_parameters(
                parameters=y_pred,
                target_scale=test_case["x"]["target_scale"],
                encoder=torch_encoder,
            )

        return metric, y_pred, y

    def _get_expected_output_shape_prediction(self, batch_size, prediction_length):
        """
        Returns the expected output shape for the prediction
        for `to_prediction`.

        Parameters
        ----------
        batch_size: int
            The size of the batch.
        prediction_length: int
            The length of the prediction.

        Returns
        -------
        tuple
            The expected output shape for the prediction.
        """

        return (batch_size, prediction_length)

    def _get_expected_output_shape_quantiles(
        self, batch_size, prediction_length, output_dim, metric_type
    ):
        """
        Returns the expected output shape for the quantiles.

        Parameters
        ----------
        batch_size: int
            The size of the batch.
        prediction_length: int
            The length of the prediction.
        output_dim: int
            The last dimension of the output tensor.
        metric_type: str
            The type of the metric (e.g., "quantile", "point_classification").
        """

        if metric_type == "point":
            return (batch_size, prediction_length, 1)
        else:
            return (batch_size, prediction_length, output_dim)

    def _test_to_prediction(self, metric, y_pred, batch_size, prediction_length):
        """Test the usage of `to_prediction` method from the metric.

        This method is used to convert the predicted values tensor into a point
        prediction. Checks the compatibility of the metric's `to_prediction` method
        with a fixed contract.

        Parameters
        ----------
        metric: Metric
            The metric instance.
        y_pred: torch.Tensor
            The predicted values tensor.
        batch_size: int
            The size of the batch. Used to determine the expected output shape.
        prediction_length: int
            The length of the prediction. Used to determine the expected output shape.
        """
        out = metric.to_prediction(y_pred)
        assert isinstance(out, torch.Tensor), "Prediction should be a tensor."
        expected_shape = self._get_expected_output_shape_prediction(
            batch_size, prediction_length
        )  # noqa: E501
        assert out.shape == expected_shape, (
            f"Prediction shape mismatch: got {out.shape}, expected {expected_shape}."  # noqa: E501
        )

    def _test_to_quantiles(
        self, metric, y_pred, batch_size, prediction_length, metric_type
    ):
        """Test the usage of `to_quantiles` method from the metric.

        This method is used to convert the predicted values tensor into quantile
        predictions. Checks the compatibility of the metric's `to_quantiles` method
        with a fixed contract.

        Parameters
        ----------
        metric: Metric
            The metric instance.
        y_pred: torch.Tensor
            The predicted values tensor.
        batch_size: int
            The size of the batch. Used to determine the expected output shape.
        prediction_length: int
            The length of the prediction. Used to determine the expected output shape.
        metric_type: str
            The type of the metric (e.g., "quantile", "point_classification").
        """
        quantiles = [0.05, 0.5, 0.95]
        output_dim = y_pred.shape[-1]

        if metric_type == "quantile" or metric_type == "point_classification":
            quantile_pred = metric.to_quantiles(y_pred)
            # for quantile metrics, the original predictions should match the result of
            # `to_quantiles`, since it does not take in the `quantiles` argument.
            assert torch.allclose(
                quantile_pred, y_pred
            ), f"Quantile prediction does not match the original predictions in {metric_type}."  # noqa: E501

            expected_shape = self._get_expected_output_shape_quantiles(
                batch_size, prediction_length, output_dim, metric_type
            )

        else:
            quantile_pred = metric.to_quantiles(y_pred, quantiles=quantiles)

            expected_shape = self._get_expected_output_shape_quantiles(
                batch_size, prediction_length, len(quantiles), metric_type
            )

        assert isinstance(
            quantile_pred, torch.Tensor
        ), "Quantile prediction should be a tensor."  # noqa: E501
        assert quantile_pred.shape == expected_shape, (
            f"Quantile prediction shape mismatch: got {quantile_pred.shape}, "
            f"expected {expected_shape}."
        )

    def _test_integration_metrics(self, metric, y_pred, y, object_pkg):
        """Test the integration of the metric with predictions and targets."""

        metric_type = object_pkg.get_class_tag("metric_type")
        assert metric_type in [
            "point",
            "point_classification",
            "quantile",
            "distribution",
        ], "Unsupported metric type for integration test."

        metric.reset()
        metric.update(y_pred, y)
        res = metric.compute()
        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Non-finite values in metric result."

        # these are pre-determined expected value for the shape.
        batch_size = y_pred.shape[0]
        prediction_length = y_pred.shape[1]

        self._test_to_prediction(metric, y_pred, batch_size, prediction_length)
        self._test_to_quantiles(
            metric, y_pred, batch_size, prediction_length, metric_type
        )
        # testing composite metrics
        composite = metric + metric
        weighted = metric * 0.5

        composite_result = composite(y_pred, y)
        weighted_result = weighted(y_pred, y)

        assert isinstance(composite_result, torch.Tensor)
        assert isinstance(weighted_result, torch.Tensor)
        assert torch.isfinite(composite_result).all()
        assert torch.isfinite(weighted_result).all()

    @pytest.mark.parametrize("target_type", ["standard", "packed", "weighted"])
    @pytest.mark.parametrize("reduction", ["mean", "none", "sqrt-mean"])
    def test_reduction_modes(
        self, object_pkg, object_instance, request, target_type, reduction
    ):  # noqa: E501
        """Test that all metrics support different reduction modes.

        The various reduction modes are ``mean``, ``none``, and ``sqrt-mean``.

        Parameters
        ----------
        object_pkg: SkbaseBaseObject
            The package object containing the metric.
        object_instance: Metric
            An instance of the metric class to be tested.
        request: pytest.FixtureRequest
            The pytest request object to access fixtures.
        target_type: str
            The type of target data (e.g., "standard", "packed", "weighted").
        reduction: str
            The reduction mode to be tested (e.g., "mean", "none", "sqrt-mean").

        Notes
        -----
        Step-outs are used to skip tests for ``sqrt-mean`` reduction mode for metrics
        that do not support it, such as those with a normal distribution type.
        """

        prepared_data = self._setup_metric_test_scenario(
            object_pkg, object_instance, target_type, request
        )

        if prepared_data is None:
            return None

        metric, y_pred, y = prepared_data

        if (
            reduction == "sqrt-mean"
            and object_pkg.get_class_tag("distribution_type") == "normal"
        ):  # noqa: E501
            return None  # sqrt-mean is not applicable for normal distribution

        metric.update(y_pred, y)
        result = metric.reduce_loss(metric.losses, metric.lengths, reduction=reduction)  # noqa: E501
        assert isinstance(result, torch.Tensor)

        if reduction == "none":
            assert result.shape == metric.losses.shape
        else:
            assert result.ndim == 0
            if reduction == "sqrt-mean":
                assert (
                    result >= 0
                ), "Result should be non-negative for sqrt-mean reduction."  # noqa: E501

    @pytest.mark.parametrize("target_type", ["standard", "packed", "weighted"])
    def test_metric_functionality(
        self, object_pkg, object_class, object_instance, request, target_type
    ):
        """Test metric functionality with appropriate test data.

        This test performs an integration test on a metric object, with the following
        steps:

        * Check if the metric supports the prediction and target types.
        * Update the metric state with predictions and targets.
        * Compute the metric value.
        * Validate usage of `to_prediction` and `to_quantiles` methods.
        * Validate the metric's ability to handle composite and weighted metrics.

        Parameters
        ----------
        object_pkg: SkbaseBaseObject
            The package object containing the metric.
        object_instance: Metric
            An instance of the metric class to be tested.
        request: pytest.FixtureRequest
            The pytest request object to access fixtures.
        target_type: str
            The type of target data (e.g., "standard", "packed", "weighted").

        Notes
        -----
        If the specific target type does not exist in the data returned by the fixture,
        the test will be skipped for that metric.
        """

        prepared_data = self._setup_metric_test_scenario(
            object_pkg, object_instance, target_type, request
        )

        if prepared_data is None:
            # meant for skipping tests for unsupported target types on certain metric
            # types
            return None

        metric, y_pred, y = prepared_data

        self._test_integration_metrics(metric, y_pred, y, object_pkg)
