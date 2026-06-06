import torch
import torch.nn as nn

from pytorch_forecasting.metrics import Metric, MultiLoss


class LossWrapper(nn.Module):
    """Unified loss wrapper for ptf-v2.

    This wrapper accepts native ptf metrics/multi-loss objects and plain
    ``torch.nn`` losses. For ``torch.nn`` losses, it automatically applies
    ``NNLossAdapter`` so users can pass ``nn.Module`` losses directly.
    """

    def __init__(self, loss: Metric | MultiLoss | nn.Module):
        super().__init__()
        if isinstance(loss, (Metric, MultiLoss)):
            self.loss = loss
        elif isinstance(loss, NNLossAdapter):
            self.loss = loss
        elif isinstance(loss, nn.Module):
            self.loss = NNLossAdapter(loss)
        else:
            raise TypeError(
                "loss must be a pytorch_forecasting Metric/MultiLoss "
                "or torch.nn.Module."
            )

    def _unwrap_target(self, target):
        if isinstance(target, (tuple, list)) and len(target) == 2 and target[1] is None:
            return target[0]
        return target

    def forward(self, y_pred, y_actual) -> torch.Tensor:
        target = self._unwrap_target(y_actual)

        # Single-target path
        if not isinstance(target, list):
            return self.loss(y_pred, y_actual)

        # Multi-target path
        if isinstance(self.loss, MultiLoss):
            return self.loss(y_pred, (target, None))

        if isinstance(self.loss, NNLossAdapter):
            losses = []
            for i, target_i in enumerate(target):
                pred_i = (
                    y_pred[i] if isinstance(y_pred, list) else y_pred[..., i : i + 1]
                )
                losses.append(self.loss(pred_i, target_i))
            return torch.stack(losses).mean()

        raise TypeError(
            "Unsupported multi-target loss setup. "
            "Use MultiLoss(...) or pass a single torch.nn loss "
            "(which is auto-wrapped and applied per target)."
        )

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        try:
            return self.loss.to_prediction(y_pred, **kwargs)
        except TypeError:
            return self.loss.to_prediction(y_pred)

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        try:
            return self.loss.to_quantiles(y_pred, **kwargs)
        except TypeError:
            return self.loss.to_quantiles(y_pred)


class NNLossAdapter(nn.Module):
    """Adapter wrapper so torch.nn losses work with ptf-v2 target/prediction shapes."""

    def __init__(self, loss_fn: nn.Module, mode: str = "auto"):
        """
        Parameters
        ----------
        loss_fn : nn.Module
            User provided torch nn loss (e.g. nn.MSELoss, nn.CrossEntropyLoss).
        mode : str
            "auto", "same_shape", or "class_index".
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.mode = mode

    def _resolve_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        if isinstance(self.loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss)):
            return "class_index"
        return "same_shape"

    def _unwrap_target(self, target):
        # target may arrive as (target, weight)
        if (
            isinstance(target, (tuple, list))
            and len(target) == 2
            and torch.is_tensor(target[0])
        ):
            return target[0]
        return target

    def forward(self, y_pred: torch.Tensor, target) -> torch.Tensor:
        target = self._unwrap_target(target)
        mode = self._resolve_mode()

        if mode == "same_shape":
            # common case: y_pred [B,T,1], target [B,T]
            if y_pred.ndim == target.ndim + 1 and y_pred.size(-1) == 1:
                y_pred = y_pred.squeeze(-1)
            if target.ndim == y_pred.ndim + 1 and target.size(-1) == 1:
                target = target.squeeze(-1)

            if y_pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {self.loss_fn.__class__.__name__}: "
                    f"y_pred={tuple(y_pred.shape)}, target={tuple(target.shape)}"
                )
            return self.loss_fn(y_pred, target)

        if mode == "class_index":
            # expected: y_pred [B,T,C], target [B,T] or [B,T,1]
            if target.ndim == y_pred.ndim and target.size(-1) == 1:
                target = target.squeeze(-1)

            n_classes = y_pred.size(-1)
            y_pred = y_pred.reshape(-1, n_classes)
            target = target.reshape(-1).long()
            return self.loss_fn(y_pred, target)

        raise ValueError(f"Unsupported adapter mode: {mode}")

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        # fallback used by BaseModel when no metric conversion is available
        if y_pred.ndim == 3 and y_pred.size(-1) == 1:
            return y_pred[..., 0]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        return y_pred
