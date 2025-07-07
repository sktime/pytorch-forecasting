"""Base object class for pytorch-forecasting metrics."""

from pytorch_forecasting.base._base_object import _BaseObject


class _BasePtMetric(_BaseObject):
    """Base class for metric object that can be discovered for testing."""

    _tags = {"object_type": "metric"}

    @classmethod
    def name(cls):
        """Get the name of the metric.

        Returns
        -------
        str
            The name of the metric.
        """
        metric_cls = cls.get_model_cls()
        return metric_cls.__name__

    @classmethod
    def get_model_cls(cls):
        """Get the metric class.

        Returns
        -------
        type
            The metric class.
        """
        raise NotImplementedError("get_metric_cls must be implemented in subclasses.")

    @classmethod
    def prepare_test_inputs(cls, test_case):
        """Prepare test inputs for the metric.

        This can be overriden by subclasses to provide special handling
        of test inputs.

        Parameters
        ----------
        test_case: dict
            Dictionary containing test case parameters.

        Returns
        -------
        (y_pred, y_actual, kwargs): tuple
            Tuple containing the predicted values, actual values, and any additional
            keyword arguments.
        """

        return test_case["y_pred"], test_case["y"]

    @classmethod
    def get_test_params(cls):
        """Returns parameters for initializing the metric for testing.

        Returns
        -------
        dict
            Dictionary containing parameters for initializing the metric.d
        """

        return {}

    def get_metric_type(cls):
        """Get the metric type."""

        return cls.get_tag("metric_type", "point")

    @classmethod
    def requires_data_type(cls):
        """Check if the metric requires a specific data type.

        Returns
        -------
        str
            The required data type.
        """
        if "metric_type" in cls._tags:
            metric_type = cls._tags["metric_type"]
        else:
            raise ValueError(
                "Metric class does not have 'metric_type' tag set. "
                "Please set the 'metric_type' tag to 'point', 'distribution', or 'quantile'."  # noqa: E501
            )

        if metric_type == "distribution":
            distribution_type = cls._tags.get("distribution_type", None)
            if distribution_type is None:
                raise ValueError(
                    "Metric requires a distribution type to be set. "
                    "Please set the 'distribution_type' tag."
                )
            return f"prepare_{distribution_type}_distribution_forecast"
        elif metric_type == "point":
            return "prepare_point_forecast"
        elif metric_type == "quantile":
            return "prepare_quantile_forecast"
        else:
            raise ValueError(
                f"Unknown metric type '{metric_type}'. "
                "Please set the 'metric_type' tag to 'point', 'distribution', or 'quantile'."  # noqa: E501
            )
