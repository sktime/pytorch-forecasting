"""
Package container for the Log Normal distribution loss metric.
"""

import torch

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric
from pytorch_forecasting.data.encoders import GroupNormalizer

class LogNormalDistributionLoss_pkg(_BasePtMetric):
    """
    Log-normal distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "log_normal",
        "info:metric_name": "LogNormalDistributionLoss",
        "requires:data_type": "log_normal_distribution_forecast",
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        },
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import LogNormalDistributionLoss

        return LogNormalDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="log")

    @classmethod
    def prepare_test_inputs(cls, test_case):
        """Prepare inputs for log normal distribution tests."""
        y_pred = test_case["y_pred"]
        y = test_case["y"]

        if isinstance(y, torch.nn.utils.rnn.PackedSequence):
            data, lengths = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
            data = torch.where(data <= 0, torch.tensor(1e-4, device=data.device), data)

            y = torch.nn.utils.rnn.pack_padded_sequence(
                data, lengths, batch_first=True, enforce_sorted=False
            )

        return y_pred, y
