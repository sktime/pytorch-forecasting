"""
Package containers for all point-forecast metrics.
"""

from pytorch_forecasting.metrics.base._base_object import _BasePtMetric


class MAE_pkg(_BasePtMetric):
    """
    Mean Average Error (MAE) metric for point forecasts.

    Defined as ``(y_pred - target).abs()``.
    """

    _tags = {"metric_type": "point", "info:metric_name": "MAE"}

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics import MAE

        return MAE


class SMAPE_pkg(_BasePtMetric):
    """
    Symmetric mean absolute percentage error metric for point forecasts.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``.
    Assumes ``y >= 0``.
    """

    _tags = {"metric_type": "point", "info:metric_name": "SMAPE"}

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.point import SMAPE

        return SMAPE


class MAPE_pkg(_BasePtMetric):
    """
    Mean absolute percentage error metric for point forecasts.

    Defined as ``(y - y_pred).abs() / y.abs()``.
    Assumes ``y >= 0``.
    """

    _tags = {"metric_type": "point", "info:metric_name": "MAPE"}

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.point import MAPE

        return MAPE


class RMSE_pkg(_BasePtMetric):
    """
    Root mean square error metric for point forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {"metric_type": "point", "info:metric_name": "RMSE"}

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.point import RMSE

        return RMSE


class PoissonLoss_pkg(_BasePtMetric):
    """
    Poisson loss for count data.

    The loss will take the exponential of the network output before it is returned as prediction.
    """  # noqa: E501

    _tags = {"metric_type": "point", "info:metric_name": "PoissonLoss"}

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.point import PoissonLoss

        return PoissonLoss


class TweedieLoss_pkg(_BasePtMetric):
    """
    Tweedie loss for regression with exponential dispersion models.

    Tweedie regression with log-link. Useful for modeling targets that might be tweedie-distributed.
    """  # noqa: E501

    _tags = {"metric_type": "point", "info:metric_name": "TweedieLoss"}

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.point import TweedieLoss

        return TweedieLoss
