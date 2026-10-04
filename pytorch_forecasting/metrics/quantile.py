"""Quantile metrics for forecasting multiple quantiles per time step."""

from typing import Optional

import torch

from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``2 * max(q * (y-y_pred), (1-q) * (y_pred-y))``

    Which is mathematically equivalent to ``2 * (q * (y-y_pred) + (|y-y_pred| - (y-y_pred))/2)``
    """  # noqa: E501

    def __init__(
        self,
        quantiles: list[float] | None = None,
        **kwargs,
    ):
        """
        Quantile loss

        Parameters
        ----------
        quantiles : list of float, optional
            quantiles for metric
        """
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        target = target.unsqueeze(-1)
        errors = target - y_pred
        q = torch.as_tensor(
            self.quantiles,
            device=y_pred.device,
            dtype=y_pred.dtype,
        )
        losses = 2 * (errors * q + (torch.abs(errors) - errors) / 2)
        return losses

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Parameters
        ----------
        y_pred : torch.Tensor
            prediction output of network

        Returns
        -------
        torch.Tensor
            point prediction
        """
        if y_pred.ndim == 3:
            if 0.5 in self.quantiles:
                idx = self.quantiles.index(0.5)
            else:
                idx = min(
                    range(len(self.quantiles)),
                    key=lambda i: abs(self.quantiles[i] - 0.5),
                )
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Parameters
        ----------
        y_pred : torch.Tensor
            prediction output of network

        Returns
        -------
        torch.Tensor
            prediction quantiles
        """
        return y_pred
