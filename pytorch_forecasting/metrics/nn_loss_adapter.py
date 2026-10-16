"""Adapter for native ``torch.nn`` loss modules as ptf metrics."""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

_Mode = Literal["point", "class", "gaussian_nll"]
_CLASS_LOSSES = (nn.CrossEntropyLoss, nn.NLLLoss)
_GAUSSIAN_NLL_LOSSES = (nn.GaussianNLLLoss,)


class NNLossAdapter(MultiHorizonMetric):
    """Adapt a ``torch.nn`` loss module to the pytorch-forecasting metric API.

    Subclasses :class:`~pytorch_forecasting.metrics.MultiHorizonMetric` so the
    wrapped loss participates in the same ecosystem as ``MAE`` / ``SMAPE``.


    Reshape mode is inferred from the wrapped loss type:

    * **point** — same-shape ``[B, T]`` after squeeze (default)
    * **class** — logits ``[B, T, C]`` vs labels ``[B, T]``
      (``CrossEntropyLoss``, ``NLLLoss``)
    * **gaussian_nll** — mean/var head ``[B, T, 2]`` (``GaussianNLLLoss``)

    Multi-target forecasting should use ``MultiLoss([NNLossAdapter(...), ...])``,
    not a single adapter over stacked targets.

    Parameters
    ----------
    loss :
        Native PyTorch loss, e.g. ``nn.MSELoss()``.
    **kwargs :
        Forwarded to :class:`~pytorch_forecasting.metrics.MultiHorizonMetric`
        (e.g. ``reduction``, ``name``).
    """

    def __init__(self, loss: nn.Module, **kwargs):
        nn_reduction = getattr(loss, "reduction", "mean")
        # MultiHorizonMetric reductions: mean | none | sqrt-mean
        if "reduction" not in kwargs:
            kwargs["reduction"] = "none" if nn_reduction == "none" else "mean"
        if "name" not in kwargs or kwargs["name"] is None:
            kwargs["name"] = f"NNLossAdapter({type(loss).__name__})"

        super().__init__(**kwargs)

        self._loss = copy.deepcopy(loss)
        if hasattr(self._loss, "reduction"):
            self._loss.reduction = "none"
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
        Literal["point", "class", "gaussian_nll"]
            ``"class"`` for ``CrossEntropyLoss`` / ``NLLLoss``,
            ``"gaussian_nll"`` for ``GaussianNLLLoss``, else ``"point"``.
        """
        if isinstance(loss, _CLASS_LOSSES):
            return "class"
        elif isinstance(loss, _GAUSSIAN_NLL_LOSSES):
            return "gaussian_nll"
        return "point"

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element losses ``[B, T]`` for :meth:`MultiHorizonMetric.update`.

        Parameters
        ----------
        y_pred :
            Network prediction (see class docstring for shapes by mode).
        target :
            Target tensor ``[B, T]`` (weight already stripped by ``update``).

        Returns
        -------
        torch.Tensor
            Unreduced loss of shape ``[B, T]``.
        """
        if isinstance(y_pred, list):
            raise ValueError(
                "NNLossAdapter does not support list of predictions "
                "with a single target tensor. For multi-target use "
                "MultiLoss([NNLossAdapter(...), ...])."
            )
        if isinstance(target, list):
            raise ValueError(
                "NNLossAdapter does not accept a list of targets. "
                "For multi-target use MultiLoss([NNLossAdapter(...), ...])."
            )

        mode = self._mode
        batch_size, time_idx = target.shape[0], target.shape[1]
        y_pred_p, target_p = self._prepare_inputs(y_pred, target, mode)
        per_elem = self._call_loss(y_pred_p, target_p, mode)

        if mode == "class":
            return per_elem.view(batch_size, time_idx)
        elif per_elem.ndim == 0:
            # defensive: some losses ignore reduction="none"
            return per_elem.expand(batch_size, time_idx)
        return per_elem

    def update(
        self,
        y_pred: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor
        | tuple[torch.Tensor, torch.Tensor | None]
        | tuple[list[torch.Tensor], torch.Tensor | None],
    ) -> None:
        """Accumulate batch loss into metric state.

        Parameters
        ----------
        y_pred : torch.Tensor or list of torch.Tensor
            Network prediction for a single target.

            * point: ``[B, T, 1]`` or ``[B, T]``
            * class: ``[B, T, C]`` logits
            * gaussian_nll: ``[B, T, 2]`` as ``(mean, raw_variance)``

            A list of prediction tensors is not supported; use
            :class:`~pytorch_forecasting.metrics.MultiLoss` for multi-target.
        target : torch.Tensor or tuple
            Ground truth. Either a tensor ``[B, T]``, or
            ``(target, weight)`` where ``weight`` is ``[B, T]`` or ``None``.
            A list of target tensors is not supported.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``target`` (or the first element of ``(target, weight)``) is a
            list of tensors.
        """
        # MultiHorizonMetric unpacks (target, weight) before calling loss();
        # catch list targets here so the message is useful.
        raw_target = target
        if (
            isinstance(target, (list, tuple))
            and len(target) == 2
            and (target[1] is None or torch.is_tensor(target[1]))
        ):
            raw_target = target[0]
        if isinstance(raw_target, list):
            raise ValueError(
                "NNLossAdapter does not accept a list of targets. "
                "For multi-target use MultiLoss([NNLossAdapter(...), ...])."
            )
        return super().update(y_pred, target)

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
            elif y_pred.size(-1) != 1:
                raise ValueError(
                    "Error inNNLossAdapter for point prediction (H=1): "
                    f"Got y_pred shape {list(y_pred.shape)} with "
                    f"H={y_pred.size(-1)}. "
                    "For multi-output / multi-target heads use MultiLoss "
                    "or a ptf metric such as QuantileLoss."
                )
            y_pred = y_pred.squeeze(-1)
            return y_pred, target

        elif mode == "class":
            if y_pred.ndim != 3:
                raise ValueError(
                    "Classification losses expect logits of shape "
                    f"(batch, time, classes), got {tuple(y_pred.shape)}."
                )
            elif target.ndim != 2:
                raise ValueError(
                    "Classification targets must have shape (batch, time), "
                    f"got {tuple(target.shape)}."
                )
            return y_pred.reshape(-1, y_pred.size(-1)), target.reshape(-1).long()

        # gaussian_nll
        else:
            if y_pred.ndim != 3 or y_pred.size(-1) != 2:
                raise ValueError(
                    "GaussianNLLLoss expects predictions of shape "
                    "(batch, time, 2) as (mean, raw_variance), "
                    f"got {tuple(y_pred.shape)}."
                )
            elif target.ndim != 2:
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
        elif mode == "class" and isinstance(self._loss, nn.NLLLoss):
            # NLLLoss expects log-probabilities
            return self._loss(F.log_softmax(y_pred, dim=-1), target)
        return self._loss(y_pred, target)

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
        elif mode == "gaussian_nll" and y_pred.ndim == 3 and y_pred.size(-1) == 2:
            return y_pred[..., 0]
        elif y_pred.ndim == 3 and y_pred.size(-1) == 1:
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
