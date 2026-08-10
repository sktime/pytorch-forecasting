"""
Moirai-MoE foundation model for pytorch-forecasting v2.
-------------------------------------------------------
"""

import math
from typing import Any
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class MoiraiMoE(TslibBaseModel):
    """
    Moirai-MoE foundation model for v2 of pytorch-forecasting.

    Moirai-MoE is a time series foundation model from Salesforce that uses a
    sparse mixture-of-experts feed-forward layer on top of a masked encoder
    Transformer, trained on billions of time points. Pretrained weights are
    loaded from the HuggingFace Hub and the model can be used zero-shot or
    fine-tuned.

    Parameters
    ----------
    loss : nn.Module
        Loss function used for training / fine-tuning.
    pretrained_model_name : str, default='Salesforce/moirai-moe-1.0-R-small'
        HuggingFace model identifier for the pretrained Moirai-MoE checkpoint.
        Common options are ``'Salesforce/moirai-moe-1.0-R-small'`` and
        ``'Salesforce/moirai-moe-1.0-R-base'``.
    patch_size : int, default=16
        Patch size used to tokenize the time series. Must be one of the patch
        sizes supported by the checkpoint.
    num_samples : int, default=100
        Number of samples drawn from the predictive distribution to produce
        the point forecast (mean over samples).
    training : bool, default=True
        If ``False`` the backbone parameters are frozen for zero-shot use.
        When ``True`` the backbone is left trainable for fine-tuning.
    logging_metrics : list[nn.Module] or None, default=None
        Metrics to log during training, validation, and testing.
    optimizer : Optimizer or str or None, default='adam'
    optimizer_params : dict or None, default=None
    lr_scheduler : str or None, default=None
    lr_scheduler_params : dict or None, default=None
    metadata : dict or None, default=None
        Metadata dictionary provided by ``TslibDataModule``.

    References
    ----------
    [1] Liu, X. et al. (2024). Moirai-MoE: Empowering Time Series Foundation
        Models with Sparse Mixture of Experts. https://arxiv.org/abs/2410.10469
    [2] https://github.com/SalesforceAIResearch/uni2ts
    [3] https://huggingface.co/Salesforce/moirai-moe-1.0-R-small

    Notes
    -----
    [1] Requires the ``uni2ts`` package (``pip install uni2ts``).
    """

    @classmethod
    def _pkg(cls):
        """Package containing the Moirai-MoE model."""
        from pytorch_forecasting.models.moirai_moe._moirai_moe_v2_pkg import (
            MoiraiMoE_pkg_v2,
        )

        return MoiraiMoE_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        pretrained_model_name: str = "Salesforce/moirai-moe-1.0-R-small",
        patch_size: int = 16,
        num_samples: int = 100,
        training: bool = True,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        warnings.warn(
            "MoiraiMoE is an experimental foundation-model adapter for ptf-v2 "
            "and depends on the unstable v2 API. Please use with caution."
        )

        self.pretrained_model_name = pretrained_model_name
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.training = training

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Load the pretrained Moirai-MoE backbone from HuggingFace."""
        try:
            from uni2ts.model.moirai_moe import MoiraiMoEModule
        except ImportError as e:
            raise ImportError(
                "The `uni2ts` package is required to use MoiraiMoE. "
                "Install it with: pip install uni2ts"
            ) from e

        self.backbone = MoiraiMoEModule.from_pretrained(self.pretrained_model_name)

        if not self.training:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _build_packed_inputs(self, history: torch.Tensor):
        """Pack a univariate context into the format Moirai-MoE expects.

        The backbone consumes patched tensors together with packing metadata
        (sample / time / variate ids and a prediction mask). Here we encode
        a single series per batch element: past values are observed, the
        prediction horizon is appended as zero placeholders and flagged in
        the prediction mask.
        """
        device = history.device
        batch, ctx, _ = history.shape
        horizon = self.prediction_length
        patch = self.patch_size

        total = ctx + horizon
        pad = (patch - total % patch) % patch
        padded = total + pad
        n_patches = padded // patch

        future = torch.zeros(
            batch, horizon + pad, 1, device=device, dtype=history.dtype
        )
        full = torch.cat([history, future], dim=1)
        target = full.view(batch, n_patches, patch)

        observed = torch.ones_like(full, dtype=torch.bool)
        observed[:, ctx:, :] = False
        observed_mask = observed.view(batch, n_patches, patch)

        prediction_mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=device)
        ctx_patches = math.ceil(ctx / patch)
        prediction_mask[:, ctx_patches:] = True

        time_id = torch.arange(n_patches, device=device).unsqueeze(0).expand(batch, -1)
        sample_id = torch.zeros(batch, n_patches, dtype=torch.long, device=device)
        variate_id = torch.zeros(batch, n_patches, dtype=torch.long, device=device)
        patch_size_t = torch.full(
            (batch, n_patches), patch, dtype=torch.long, device=device
        )

        return {
            "target": target,
            "observed_mask": observed_mask,
            "sample_id": sample_id,
            "time_id": time_id,
            "variate_id": variate_id,
            "prediction_mask": prediction_mask,
            "patch_size": patch_size_t,
            "ctx_patches": ctx_patches,
        }

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the foundation model on a single univariate series per batch."""
        if "history_target" in x and x["history_target"].size(-1) > 0:
            history = x["history_target"][..., :1]
        elif "history_cont" in x and x["history_cont"].size(-1) > 0:
            history = x["history_cont"][..., :1]
        else:
            raise ValueError(
                "MoiraiMoE expects `history_target` or `history_cont` in the "
                "input batch."
            )

        means = history.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(history, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normed = (history - means) / stdev

        packed = self._build_packed_inputs(normed)

        distr = self.backbone(
            target=packed["target"],
            observed_mask=packed["observed_mask"],
            sample_id=packed["sample_id"],
            time_id=packed["time_id"],
            variate_id=packed["variate_id"],
            prediction_mask=packed["prediction_mask"],
            patch_size=packed["patch_size"],
        )

        samples = distr.sample(torch.Size([self.num_samples]))
        point = samples.mean(dim=0)

        batch = history.shape[0]
        flat = point.reshape(batch, -1)
        ctx_end = packed["ctx_patches"] * self.patch_size
        pred = flat[:, ctx_end : ctx_end + self.prediction_length]

        pred = pred * stdev.squeeze(-1) + means.squeeze(-1)
        return pred.unsqueeze(-1)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of Moirai-MoE.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary produced by ``TslibDataModule``. At minimum,
            ``history_target`` or ``history_cont`` must be present.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"prediction": tensor}`` with shape
            ``(batch, prediction_length, target_dim)``.
        """
        prediction = self._forecast(x)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
