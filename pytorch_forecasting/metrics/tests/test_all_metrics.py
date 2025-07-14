"""Automated test for all metrics in PyTorch Forecasting."""

import pytest
import torch

from pytorch_forecasting.tests.test_all_estimators import (
    BaseFixtureGenerator,
    PackageConfig,
)


class TestAllPtMetrics(PackageConfig, BaseFixtureGenerator):
    """Test suite for all metrics in PyTorch Forecasting."""

    object_type_filter = "metric"

    fixture_sequence = ["object_pkg", "object_class"]

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

    def _test_integration_metrics(self, metric, y_pred, y, object_pkg):
        metric.reset()
        metric.update(y_pred, y)
        res = metric.compute()
        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Non-finite values in metric result."

        point_pred = metric.to_prediction(y_pred)
        assert isinstance(point_pred, torch.Tensor), "Prediction should be a tensor."
        assert (
            point_pred.shape[:2] == y_pred.shape[:2]
        ), "Prediction shape mismatch with y_pred."  # noqa: E501
        assert point_pred.ndim == 2
        if object_pkg.get_class_tag("info:metric_name") == "CrossEntropy":
            assert torch.all(
                point_pred.ge(0)
            ), "CrossEntropy loss predictions should be non-negative."  # noqa: E501

        quantiles = [0.1, 0.5, 0.9]
        if object_pkg.get_class_tag("metric_type") == "quantile":
            quantile_pred = metric.to_quantiles(y_pred)
            assert isinstance(
                quantile_pred, torch.Tensor
            ), "Quantile prediction should be a tensor."  # noqa: E501
            assert quantile_pred.shape == y_pred.shape
        else:
            quantile_pred = metric.to_quantiles(y_pred, quantiles=quantiles)
            assert isinstance(
                quantile_pred, torch.Tensor
            ), "Quantile prediction should be a tensor."  # noqa: E501

            # capability:quantile_generation is a tag for metrics
            # that override the default `to_quantiles` method from `Metric` base class.
            if y_pred.ndim == 3 or (
                object_pkg.get_class_tag("shape:adds_quantile_dimension")
            ):
                assert quantile_pred.shape == (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    len(quantiles),
                ), f"`to_quantiles()`, fails for metric {metric}()."
            elif y_pred.ndim == 2:
                assert quantile_pred.shape == (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    1,
                ), f"`to_quantiles()`, fails for metric {metric}()."

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
