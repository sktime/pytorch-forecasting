import torch
import torch.nn as nn


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

