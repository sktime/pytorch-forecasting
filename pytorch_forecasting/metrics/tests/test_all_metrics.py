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

    def _check_required_dependencies(self, object_pkg):
        """
        Skip tests if the required dependencies for the metric are not installed
        in your environment.
        """

        required_deps = object_pkg.get_class_tag("python_dependencies")
        class_name = object_pkg.name()
        if required_deps:
            try:
                # Use the dependency checking utility
                if not _check_soft_dependencies(required_deps, severity="none"):
                    pytest.skip(
                        f"Skipping test for {class_name} - missing dependencies: {required_deps}"  # noqa: E501
                    )
                    return False
            except Exception as e:
                # Catch any unexpected errors in dependency checking
                pytest.skip(f"Error checking dependencies for {class_name}: {str(e)}")
                return False
        return True

    def _generate_object_instance(self, test_name, **kwargs):
        """Generate instance of the metric class for testing parametrized
        with scenarios in get_metric_test_params().
        """

        if "object_pkg" in kwargs:
            obj_meta = kwargs["object_pkg"]
        else:
            return []

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

        if not self._check_required_dependencies(object_pkg):
            return None

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

    def _test_to_prediction(self, metric, y_pred):
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
        """
        out = metric.to_prediction(y_pred)
        assert isinstance(out, torch.Tensor), "Prediction should be a tensor."
        assert out.ndim == 2, "Prediction should be a 2D tensor."
        if y_pred.ndim == 2:
            assert out.shape == y_pred.shape, "Prediction shape mismatch with y_pred."  # noqa: E501
        elif y_pred.ndim == 3 and metric.quantiles is None:
            assert out.shape == (
                y_pred.shape[0],
                getattr(metric, "prediction_length", y_pred.shape[1]),
            )  # noqa: E501
        elif y_pred.ndim == 3 and metric.quantiles is not None:
            # case for quantile loss.
            assert out.shape == (y_pred.shape[0], y_pred.shape[1])
            assert 0.5 in metric.quantiles, (
                "`metric.quantiles` should include the median quantile",
                "for point prediction.",
            )
        else:
            raise AssertionError("Unhandled y_pred shape for `to_prediction` method.")  # noqa: E501

    def _test_to_quantiles(self, object_pkg, metric, y_pred):
        """Test the usage of `to_quantiles` method from the metric.

        This method is used to convert the predicted values tensor into quantile
        predictions. Checks the compatibility of the metric's `to_quantiles` method
        with a fixed contract.

        Parameters
        ----------
        object_pkg: SkbaseBaseObject
            The package object containing the metric.
        metric: Metric
            The metric instance.
        y_pred: torch.Tensor
            The predicted values tensor.
        """
        metric_type = object_pkg.get_class_tag("metric_type")
        quantiles = [0.05, 0.5, 0.95]
        if metric_type == "quantile" or metric_type == "point_classification":
            quantile_pred = metric.to_quantiles(y_pred)
            # for quantile metrics, the original predictions should match the result of
            # `to_quantiles`, since it does not take in the `quantiles` argument.
            assert torch.allclose(
                quantile_pred, y_pred
            ), f"Quantile prediction does not match the original predictions in {object_pkg.name()}."  # noqa: E501

        else:
            quantile_pred = metric.to_quantiles(y_pred, quantiles=quantiles)

        assert isinstance(
            quantile_pred, torch.Tensor
        ), "Quantile prediction should be a tensor."  # noqa: E501
        assert (
            quantile_pred.shape[0] == y_pred.shape[0]
        ), "Batch size mismatch between quantiles."  # noqa: E501
        assert (
            quantile_pred.shape[1] == y_pred.shape[1]
        ), "Sequence length mismatch between quantiles."  # noqa: E501
        if y_pred.ndim == 2:
            # 2D input: output should be (batch, time, 1)
            quantile_dim = 1
        elif y_pred.ndim == 3:
            if object_pkg.get_class_tag("metric_type") == "quantile":
                # Quantile metric: output should match input's quantile dim
                quantile_dim = y_pred.shape[2]
            else:
                # All other metrics: output should match number of quantiles requested
                quantile_dim = len(quantiles)
        else:
            raise AssertionError(f"Unhandled y_pred shape: {y_pred.shape}")
        assert (
            quantile_pred.shape[2] == quantile_dim
        ), f"Quantile prediction shape mismatch: got {quantile_pred.shape}, expected last dim {quantile_dim}."  # noqa: E501

    def _test_integration_metrics(self, metric, y_pred, y, object_pkg):
        metric.reset()
        metric.update(y_pred, y)
        res = metric.compute()
        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Non-finite values in metric result."

        self._test_to_prediction(metric, y_pred)
        self._test_to_quantiles(object_pkg, metric, y_pred)

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
