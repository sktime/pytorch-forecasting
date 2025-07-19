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

    fixture_sequence = ["object_pkg", "object_class"]

    pass


class TestAllPtMetrics(MetricPackageConfig, MetricFixtureGenerator):
    """Test suite for all metrics in PyTorch Forecasting."""

    object_type_filter = "metric"

    def _setup_metric_test_scenario(
        self,
        object_pkg,
        object_class,
        target_type,
        request,
    ):
        """Prepare test inputs for the given metric.

        Parameters
        ----------
        object_pkg: SkbaseBaseObject
            The package object containing the metric.
        object_class: class
            The class of the metric to be tested.
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

        required_deps = object_pkg.get_class_tag("python_dependencies")
        if required_deps:
            try:
                # Use the dependency checking utility
                if not _check_soft_dependencies(required_deps, severity="none"):
                    pytest.skip(
                        f"Skipping test for {object_class.__name__} - missing dependencies: {required_deps}"  # noqa: E501
                    )
                    return None
            except Exception as e:
                # Catch any unexpected errors in dependency checking
                pytest.skip(
                    f"Error checking dependencies for {object_class.__name__}: {str(e)}"
                )  # noqa: E501
                return None

        prepare_data_fixture_name = object_pkg.requires_data_type()
        data = request.getfixturevalue(prepare_data_fixture_name)

        if target_type not in data["test_cases"]:
            return None

        test_case = data["test_cases"][target_type]
        y_pred, y = object_pkg.prepare_test_inputs(test_case)

        test_params = object_pkg.get_test_params()
        metric = object_class(**test_params)

        torch_encoder = object_pkg.get_encoder()
        if not object_pkg.get_class_tag("no_rescaling"):
            y_pred = metric.rescale_parameters(
                parameters=y_pred,
                target_scale=test_case["x"]["target_scale"],
                encoder=torch_encoder,
            )

        return metric, y_pred, y

    def _test_to_quantile_on_quantile_metrics(
        self,
        y_pred,
        quantile_pred,
    ):
        """Test the usage of `to_quantiles` in quantile loss metrics. This covers usage
        in the context of the current state of the library which only includes
        `QuantileLoss`.

        `QuantileLoss` simply returns the input `y_pred` as is and gives no
        consideration to the required quantiles. This is because the
        `y_pred` is already a 3d tensor with the 3rd dimension containing the quantiles.

        Parameters
        ----------
        metric: QuantileLoss
            The quantile loss metric instance.
        y_pred: torch.Tensor
            The predicted values tensor.
        quantiles: list
            A list of quantiles to be tested.
        """
        assert isinstance(
            quantile_pred, torch.Tensor
        ), "Quantile prediction should be a tensor."
        assert quantile_pred.shape == y_pred.shape

    def _test_to_quantile_on_point_metrics(
        self, metric, y_pred, quantile_pred, quantiles
    ):
        """Test the usage of `to_quantiles` method from base metrics. This is normally
        used for point predictions which do no override the `to_quantiles`. An exception
        to this is the `PoissonLoss` which has its own implementation of `to_quantiles`.

        This specifically tests the case where y_pred is 3d tensor with size(2) > 1,
        which replaces the values of the 3rd dimension with the quantiles generated by
        a call to the torch.quantile method in the `to_quantiles` method of the
        `Metric` base class. We assert the equality of the quantile predictions.

        Parameters
        ----------
        metric: Metric
            The metric instance.
        y_pred: torch.Tensor
            The predicted values tensor.
        quantile_pred: torch.Tensor
            The quantile predictions tensor from the metric's `to_quantiles` method.
        quantiles: list
            A list of quantiles to be tested.
        """
        if y_pred.ndim == 2:
            assert quantile_pred.shape == (
                y_pred.shape[0],
                y_pred.shape[1],
                1,
            ), f"`to_quantiles()`, fails for metric {metric}."
        elif y_pred.ndim == 3 and y_pred.size(2) > 1:
            expected_pred = torch.quantile(
                y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2
            ).permute(1, 2, 0)

            torch.testing.assert_close(
                quantile_pred,
                expected_pred,
                msg=f"Quantile predictions {quantile_pred} \n do not match expected values \n {expected_pred}",  # noqa: E501
            )

    def _test_to_quantile_custom(self, metric, y_pred, quantile_pred, quantiles):
        """Test the usage of `to_quantiles` method for metrics that override
        the default behavior in `Metric` base class.

        The test does not attempt to replicate the internal implementation of
        quantile prediction generation inside the metric class and equate it with the
        provided `quantile_pred` since the generation of quantiles in case of
        distribution losses is a stochastic process, and the quantile prediction
        generated on the distribution data may never match with our `quantile_pred`
        tensor.

        Parameters
        ----------
        metric: Metric
            The metric instance that overrides `to_quantiles`.
        y_pred: torch.Tensor
            The predicted values tensor.
        quantiles: list
            A list of quantiles to be tested.
        """

        # the reason we have a separate case of mqf2 is because
        # it produces outputs from `to_quantiles` only on the
        # input `prediction_length` and not on the entire sequence
        # length of the `y_pred` tensor.

        assert quantile_pred.shape == (
            y_pred.shape[0],
            y_pred.shape[1],
            len(quantiles),
        ), f"`to_quantiles()`, fails for metric {metric}."

    def _test_to_quantile_mqf2(self, metric, y_pred, quantile_pred, quantiles):
        """Test the usage of `to_quantiles` method for MQF2 distribution loss metric.

        The test does not attempt to replicate the internal implementation of
        quantile prediction generation inside the metric class and equate it with the
        provided `quantile_pred` since the generation of quantiles in case of
        distribution losses is a stochastic process, and the quantile prediction
        generated on the distribution data may never match with our `quantile_pred`
        tensor.

        Parameters
        ----------
        metric: Metric
            The metric instance that overrides `to_quantiles`.
        y_pred: torch.Tensor
            The predicted values tensor.
        quantiles: list
            A list of quantiles to be tested.
        """

        # the reason we have a separate case of mqf2 is because
        # it produces outputs from `to_quantiles` only on the
        # input `prediction_length` and not on the entire sequence
        # length of the `y_pred` tensor.

        assert quantile_pred.shape == (
            y_pred.shape[0],
            metric.prediction_length,
            len(quantiles),
        ), f"`to_quantiles()`, fails for metric {metric}."

    def _test_integration_metrics(self, metric, y_pred, y, object_pkg):
        metric.reset()
        metric.update(y_pred, y)
        res = metric.compute()
        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Non-finite values in metric result."

        point_pred = metric.to_prediction(y_pred)
        assert isinstance(point_pred, torch.Tensor), "Prediction should be a tensor."
        if object_pkg.get_class_tag("info:metric_name") == "MQF2DistributionLoss":
            # MQF2 produces outputs from `to_prediction`
            # only for the specified `prediction_length` (i.e., the forecast horizon),
            # rather than for the entire input sequence length of y_pred.
            # Hence, the separate case.
            assert (
                point_pred.shape[0] == y_pred.shape[0]
            ), "Prediction batch size mismatch with y_pred."
            assert (
                point_pred.shape[1] == metric.prediction_length
            ), "Prediction length mismatch with metric's expected prediction length."
        else:
            assert (
                point_pred.shape[:2] == y_pred.shape[:2]
            ), "Prediction shape mismatch with y_pred."  # noqa: E501
        assert point_pred.ndim == 2
        if object_pkg.get_class_tag("info:metric_name") == "CrossEntropy":
            assert torch.all(
                point_pred.ge(0)  # expects all class probabilities to be non-negative.
            ), "CrossEntropy loss predictions should be non-negative."  # noqa: E501

        quantiles = [0.1, 0.5, 0.9]

        # Calculate quantile predictions once for all tests
        # Determine metric type and test accordingly
        metric_type = object_pkg.get_class_tag("metric_type")
        has_custom_quantiles = object_pkg.get_class_tag(
            "capability:quantile_generation"
        )  # noqa: E501

        if metric_type != "quantile":
            quantile_pred = metric.to_quantiles(y_pred, quantiles=quantiles)
            assert isinstance(
                quantile_pred, torch.Tensor
            ), "Quantile prediction should be a tensor."  # noqa: E501

        if metric_type == "quantile":
            # quantile loss does not accept quantiles as input
            quantile_pred = metric.to_quantiles(y_pred)
            assert isinstance(
                quantile_pred, torch.Tensor
            ), "Quantile prediction should be a tensor."  # noqa: E501
            # Quantile metrics just return the input tensor
            self._test_to_quantile_on_quantile_metrics(y_pred, quantile_pred)
        elif metric_type == "distribution" or has_custom_quantiles:
            # Distribution metrics and some point metrics have their own implementation.
            if object_pkg.get_class_tag("distribution_type") == "mqf2":
                # MQF2DistributionLoss has its own implementation of `to_quantiles`
                # which is different from the base `Metric` class.
                # It produces outputs only for the specified `prediction_length`
                # (i.e., the forecast horizon), rather than for the entire input
                # sequence length of the `y_pred` tensor. Hence, the separate case.
                self._test_to_quantile_mqf2(metric, y_pred, quantile_pred, quantiles)
            else:
                self._test_to_quantile_custom(metric, y_pred, quantile_pred, quantiles)
        else:
            if object_pkg.get_class_tag("info:metric_name") == "CrossEntropy":
                # CrossEntropyLoss does not support quantiles
                return None

            # Standard point metrics using the base implementation
            self._test_to_quantile_on_point_metrics(
                metric, y_pred, quantile_pred, quantiles
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
        self, object_pkg, object_class, request, target_type, reduction
    ):  # noqa: E501
        """Test that all metrics support different reduction modes.

        The various reduction modes are ``mean``, ``none``, and ``sqrt-mean``.

        Parameters
        ----------
        object_pkg: SkbaseBaseObject
            The package object containing the metric.
        object_class: class
            The class of the metric to be tested.
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
            object_pkg, object_class, target_type, request
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
    def test_metric_functionality(self, object_pkg, object_class, request, target_type):
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
        object_class: class
            The class of the metric to be tested.
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
            object_pkg, object_class, target_type, request
        )

        if prepared_data is None:
            # meant for skipping tests for unsupported target types on certain metric
            # types
            return None

        metric, y_pred, y = prepared_data

        self._test_integration_metrics(metric, y_pred, y, object_pkg)
