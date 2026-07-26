"""Adapter for native ``torch.nn`` loss modules in ptf-v2."""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_Mode = Literal["point", "class", "gaussian_nll"]
_CLASS_LOSSES = (nn.CrossEntropyLoss, nn.NLLLoss)
_GAUSSIAN_NLL_LOSSES = (nn.GaussianNLLLoss,)


class NNLossAdapter(nn.Module):
    """Adapt a ``torch.nn`` loss module to the ptf-v2 loss API.

    Wraps a standard PyTorch loss (nn.Module) to handle the specific
    input formats used in pytorch-forecasting v2, such as (target, weight) tuples
    and multi-target list of tensors.

    The reshape mode is inferred automatically from the wrapped loss type:

    * **point** — same-shape ``[B, T]`` after squeeze (default)
    * **class** — logits ``[B, T, C]`` vs labels ``[B, T]``
      (``CrossEntropyLoss``, ``NLLLoss``)
    * **gaussian_nll** — mean/var head ``[B, T, 2]`` (``GaussianNLLLoss``)

    Parameters
    ----------
    loss :
        Native PyTorch loss, e.g. ``nn.MSELoss()``.
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        # deepcopy so we never mutate the caller's loss instance
        self._loss = copy.deepcopy(loss)
        self._reduction = getattr(loss, "reduction", "mean")
        self._mode = self._infer_mode(self._loss)

    @staticmethod
    def _infer_mode(loss: nn.Module) -> _Mode:
        """Map a native loss class to the adapter reshape/call mode.

        Parameters
        ----------
        loss :
            Wrapped ``torch.nn`` loss module.

        Returns
        -------
        {"point", "class", "gaussian_nll"}
            ``"class"`` for ``CrossEntropyLoss`` / ``NLLLoss``,
            ``"gaussian_nll"`` for ``GaussianNLLLoss``, else ``"point"``.
        """
        if isinstance(loss, _CLASS_LOSSES):
            return "class"
        if isinstance(loss, _GAUSSIAN_NLL_LOSSES):
            return "gaussian_nll"
        return "point"

    def forward(
        self,
        y_pred: torch.Tensor | list[torch.Tensor],
        y_actual: torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor | list[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of the adapter.

        Parameters
        ----------
        y_pred :
            Model predictions.

            * point: ``[B, T, 1]`` / ``[B, T]``, or ``[B, T, N]`` for multi-target
            * class: ``[B, T, C]`` logits
            * gaussian_nll: ``[B, T, 2]`` as ``(mean, raw_variance)``
        y_actual :
            Targets, optionally as ``(target, weight)``. Multi-target uses a
            list of ``[B, T]`` tensors (point mode only).

        Returns
        -------
        torch.Tensor
            Scalar (or unreduced) loss.
        """
        target, weight = self._unpack_y_actual(y_actual)
        mode = self._mode

        # multi-target scenario
        if isinstance(target, list):
            if mode != "point":
                raise ValueError(
                    f"Error in NNLossAdapter: Multi-target lists are only supported"
                    f"for point losses, got {self._loss.__class__.__name__!r}."
                )
            if not isinstance(y_pred, torch.Tensor):
                raise ValueError(
                    f"NNLossAdapter expected y_pred to be a torch.Tensor for "
                    f"multi-target, but got {type(y_pred)}. Standard multi-target "
                    f"in ptf-v2 expects y_pred of shape [B, T, N]."
                )
            # y_pred is [B, T, N], split along last dimension
            y_preds = [yp.squeeze(-1) for yp in y_pred.split(1, dim=-1)]
            if len(y_preds) != len(target):
                raise ValueError(
                    f"Number of predictions ({len(y_preds)}) does not match "
                    f"number of targets ({len(target)})."
                )

            total_loss = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            for yp, t in zip(y_preds, target):
                total_loss = total_loss + self._compute_loss(yp, t, weight, mode)
            return total_loss

        # single-target scenario
        if isinstance(y_pred, list):
            raise ValueError(
                "NNLossAdapter does not support list of predictions "
                "with single target tensor."
            )

        y_pred, target = self._prepare_inputs(y_pred, target, mode)
        return self._compute_loss(y_pred, target, weight, mode)

    @staticmethod
    def _unpack_y_actual(
        y_actual: torch.Tensor | list[torch.Tensor] | tuple,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | None]:
        """Split ``y_actual`` into ``(target, weight)``.

        Accepts a bare target tensor/list, ``(target, weight)``,
        ``(target, None)``, or a length-1 sequence wrapping the target only.

        Parameters
        ----------
        y_actual :
            Target payload from the ptf-v2 training loop.

        Returns
        -------
        target :
            Tensor or list of target tensors.
        weight :
            Optional sample/time weight tensor, or ``None``.
        """
        if (
            isinstance(y_actual, tuple)
            and len(y_actual) == 2
            and torch.is_tensor(y_actual[1])
        ):
            return y_actual[0], y_actual[1]
        # also allow (target, None) or len-2 list/tuple without weight tensor
        if isinstance(y_actual, (list, tuple)) and not isinstance(
            y_actual, torch.Tensor
        ):
            if len(y_actual) == 2 and (
                y_actual[1] is None or torch.is_tensor(y_actual[1])
            ):
                return y_actual[0], y_actual[1]
            if len(y_actual) == 1:
                return y_actual[0], None
        return y_actual, None

    def _prepare_inputs(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mode: _Mode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape ``y_pred`` / ``target`` into the layout expected by ``mode``.

        * **point** — squeeze trailing singleton ``H=1`` so both are ``[B, T]``
        * **class** — flatten to ``(B*T, C)`` logits and ``(B*T,)`` long labels
        * **gaussian_nll** — keep ``[B, T, 2]`` preds and ``[B, T]`` targets

        Parameters
        ----------
        y_pred :
            Raw network prediction tensor.
        target :
            Raw target tensor (single-target path only).
        mode :
            Adapter mode from :meth:`_infer_mode`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Prepared ``(y_pred, target)`` ready for :meth:`_call_loss`.
        """
        if mode == "point":
            if y_pred.ndim != 3:
                return y_pred, target
            if y_pred.size(-1) != 1:
                raise ValueError(
                    f"NNLossAdapter only supports point predictions (H=1). "
                    f"Got y_pred shape {list(y_pred.shape)} with "
                    f"H={y_pred.size(-1)}. "
                    "For multi-horizon losses, use a ptf metrics loss instead."
                )
            y_pred = y_pred.squeeze(-1)
            return y_pred, target

        elif mode == "class":
            if y_pred.ndim != 3:
                raise ValueError(
                    "Classification losses expect logits of shape "
                    f"(batch, time, classes), got {tuple(y_pred.shape)}."
                )
            if target.ndim != 2:
                raise ValueError(
                    "Classification targets must have shape (batch, time), "
                    f"got {tuple(target.shape)}."
                )
            return y_pred.reshape(-1, y_pred.size(-1)), target.reshape(-1).long()

        # gaussian_nll
        if y_pred.ndim != 3 or y_pred.size(-1) != 2:
            raise ValueError(
                "GaussianNLLLoss expects predictions of shape "
                f"(batch, time, 2) as (mean, raw_variance); got {tuple(y_pred.shape)}."
            )
        if target.ndim != 2:
            raise ValueError(
                "GaussianNLL targets must have shape (batch, time), "
                f"got {tuple(target.shape)}."
            )
        return y_pred, target

    def _call_loss(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mode: _Mode,
    ) -> torch.Tensor:
        """Invoke the wrapped ``torch.nn`` loss with mode-specific arguments.

        * **gaussian_nll** — split mean / softplus(raw_var)+eps, call
          ``loss(mean, target, var)``
        * **class** + ``NLLLoss`` — apply ``log_softmax`` before the loss
        * otherwise — ``loss(y_pred, target)``

        Parameters
        ----------
        y_pred :
            Prepared prediction tensor from :meth:`_prepare_inputs`.
        target :
            Prepared target tensor.
        mode :
            Adapter mode controlling the call signature.

        Returns
        -------
        torch.Tensor
            Loss as returned by the wrapped module (honoring its reduction,
            unless temporarily overridden by :meth:`_compute_loss`).
        """
        if mode == "gaussian_nll":
            mean = y_pred[..., 0]
            var = F.softplus(y_pred[..., 1]) + 1e-6
            return self._loss(mean, target, var)
        if mode == "class" and isinstance(self._loss, nn.NLLLoss):
            # NLLLoss expects log-probabilities
            return self._loss(F.log_softmax(y_pred, dim=-1), target)
        return self._loss(y_pred, target)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        mode: _Mode,
    ) -> torch.Tensor:
        """Compute (optionally sample-weighted) loss for one target.

        When ``weight`` is set, temporarily forces ``reduction="none"``,
        multiplies elementwise, then reduces with the original reduction
        (weighted mean via ``sum(loss*w)/sum(w)``, ``sum``, or unreduced).

        Parameters
        ----------
        y_pred :
            Prepared prediction tensor.
        target :
            Prepared target tensor.
        weight :
            Optional weights broadcastable to the unreduced loss.
        mode :
            Used to flatten class-mode weights to ``(B*T,)``.

        Returns
        -------
        torch.Tensor
            Scalar or unreduced weighted loss.
        """
        if weight is None:
            return self._call_loss(y_pred, target, mode)

        old_reduction = getattr(self._loss, "reduction", None)
        if old_reduction is not None:
            self._loss.reduction = "none"
        try:
            loss = self._call_loss(y_pred, target, mode)
        finally:
            if old_reduction is not None:
                self._loss.reduction = old_reduction

        # class mode flattens to (B*T,); flatten matching weights
        if mode == "class" and weight is not None:
            weight = weight.reshape(-1)
        elif mode == "gaussian_nll" and weight is not None and weight.ndim == 2:
            pass  # already [B, T], matches per-element loss

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
        # 'none' or others
        return weighted_loss

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert network output to a point forecast.

        * **class** — ``argmax`` over the class dimension
        * **gaussian_nll** — mean channel (index 0)
        * **point** — squeeze trailing ``H=1`` when present

        Parameters
        ----------
        y_pred :
            Raw network prediction tensor.
        **kwargs :
            Accepted for API compatibility; ignored.

        Returns
        -------
        torch.Tensor
            Point prediction, typically ``[B, T]``.
        """
        del kwargs
        mode = self._mode
        if mode == "class" and y_pred.ndim == 3:
            return y_pred.argmax(dim=-1)
        if mode == "gaussian_nll" and y_pred.ndim == 3 and y_pred.size(-1) == 2:
            return y_pred[..., 0]
        if y_pred.ndim == 3 and y_pred.size(-1) == 1:
            return y_pred.squeeze(-1)
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """Expose a quantile-shaped view of the point prediction.

        Native ``torch.nn`` losses are not quantile models; this wraps
        :meth:`to_prediction` and adds a trailing singleton dim when needed
        so callers expecting ``[B, T, Q]`` still work.

        Parameters
        ----------
        y_pred :
            Raw network prediction tensor.
        **kwargs :
            Accepted for API compatibility; ignored.

        Returns
        -------
        torch.Tensor
            ``[B, T, 1]`` when the point forecast is ``[B, T]``, else the
            point forecast unchanged.
        """
        del kwargs
        point = self.to_prediction(y_pred)
        if point.ndim == 2:
            return point.unsqueeze(-1)
        return point
