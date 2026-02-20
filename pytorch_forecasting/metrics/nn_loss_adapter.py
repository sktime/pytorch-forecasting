from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class NNLossAdapter(nn.Module):
    """Adapter to use PyTorch nn losses in ptf-v2.

    This class wraps a standard PyTorch loss (nn.Module) to handle the specific
    input formats used in pytorch-forecasting v2, such as (target, weight) tuples
    and multi-target list of tensors.

    Args:
        loss (nn.Module): The PyTorch loss to wrap.
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(
        self,
        y_pred: Union[torch.Tensor, List[torch.Tensor]],
        y_actual: Union[torch.Tensor, Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass of the adapter.

        Args:
            y_pred (Union[torch.Tensor, List[torch.Tensor]]): Model predictions.
                Expected to be [B, T, N] for multi-target or [B, T, 1] / [B, T] for single target.
            y_actual (Union[torch.Tensor, Tuple]): Actual values and optionally weights.
                Can be a tensor, or a tuple (target, weight), where target can be a list of tensors.

        Returns:
            torch.Tensor: The computed and reduced loss.
        """
        # Handle y_actual as (target, weight) or just target
        if isinstance(y_actual, (list, tuple)) and not isinstance(y_actual, torch.Tensor):
            if len(y_actual) == 2:
                target, weight = y_actual
            else:
                target = y_actual[0]
                weight = None
        else:
            target, weight = y_actual, None

        if isinstance(target, list):
            # Multi-target scenario
            if not isinstance(y_pred, torch.Tensor):
                raise ValueError(
                    f"NNLossAdapter expected y_pred to be a torch.Tensor for multi-target, "
                    f"but got {type(y_pred)}. Standard multi-target in ptf-v2 expects "
                    f"y_pred of shape [B, T, N]."
                )

            # y_pred is [B, T, N], split along last dimension
            y_preds = y_pred.split(1, dim=-1)
            y_preds = [yp.squeeze(-1) for yp in y_preds]

            if len(y_preds) != len(target):
                raise ValueError(
                    f"Number of predictions ({len(y_preds)}) does not match "
                    f"number of targets ({len(target)})."
                )

            total_loss = torch.tensor(0.0, device=y_pred.device)
            for yp, t in zip(y_preds, target):
                total_loss = total_loss + self._compute_loss(yp, t, weight)
            return total_loss
        else:
            # Single target scenario
            if isinstance(y_pred, list):
                # Error if list of predictions but single tensor target
                raise ValueError(
                    "NNLossAdapter does not support list of predictions with single target tensor."
                )

            if y_pred.ndim == 3:
                if y_pred.size(-1) != 1:
                    raise ValueError(
                        f"NNLossAdapter only supports point predictions (H=1). "
                        f"Got y_pred shape {list(y_pred.shape)} with H={y_pred.size(-1)}. "
                        "For multi-horizon losses, use a ptf metrics loss instead."
                    )
                y_pred = y_pred.squeeze(-1)

            return self._compute_loss(y_pred, target, weight)

    def _compute_loss(
        self, y_pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss for a single target, applying weights if provided.
        """
        if weight is None:
            return self.loss(y_pred, target)

        # Handle weighting
        old_reduction = getattr(self.loss, "reduction", "mean")
        self.loss.reduction = "none"
        try:
            loss = self.loss(y_pred, target)
        finally:
            self.loss.reduction = old_reduction

        # Ensure weight has same dimensions as loss for multiplication
        if weight.ndim < loss.ndim:
            weight = weight.unsqueeze(-1).expand_as(loss)
        elif weight.ndim > loss.ndim:
            # Squeeze weight if it has more dimensions (e.g. [B, T, 1] vs [B, T])
            weight = weight.squeeze(-1)

        weighted_loss = loss * weight

        if old_reduction == "mean":
            return weighted_loss.sum() / weight.sum()
        elif old_reduction == "sum":
            return weighted_loss.sum()
        else:
            # 'none' or others
            return weighted_loss

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.
        """
        if y_pred.ndim == 3:
            if y_pred.size(-1) == 1:
                return y_pred.squeeze(-1)
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.
        """
        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        return y_pred
