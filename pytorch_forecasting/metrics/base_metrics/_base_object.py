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
        metric_cls = cls.get_cls()
        return metric_cls.__name__

    @classmethod
    def get_cls(cls):
        """Get the metric class.

        Returns
        -------
        type
            The metric class.
        """
        raise NotImplementedError("get_cls must be implemented in subclasses.")

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
    def get_metric_test_params(cls):
        """Returns parameters for initializing the metric for testing.

        Returns
        -------
        dict
            Dictionary containing parameters for initializing the metric.d
        """

        return []

    @classmethod
    def get_encoder(cls):
        """Get the encoder for the metric.

        This can be overridden by subclasses to provide a specific encoder.

        Returns
        -------
        TorchNormalizer
            An instance of TorchNormalizer or similar encoder.
        """
        from pytorch_forecasting.data import TorchNormalizer

        return TorchNormalizer()
