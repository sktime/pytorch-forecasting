"""Zero-shot Moirai-MoE forecasts through the optional uni2ts package."""

from skbase.utils.dependencies import _check_soft_dependencies
import torch
from torch import nn

from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class MoiraiMoE(TslibBaseModel):
    """Salesforce Moirai-MoE zero-shot forecaster.

    Parameters
    ----------
    model_id : str, default="Salesforce/moirai-moe-1.0-R-small"
        Hugging Face repository or local directory containing pretrained weights.
        Ignored when ``module`` is provided.
    module : nn.Module or None, default=None
        An initialized ``uni2ts.model.moirai_moe.MoiraiMoEModule`` backbone.
        This allows inference with local weights without a Hub download.
    num_samples : int, default=100
        Number of forecast samples used for means and empirical quantiles.
    patch_size : int, default=16
        Patch size supported by the supplied backbone.
    metadata : dict
        Metadata from ``TslibDataModule``, including ``context_length``,
        ``prediction_length``, and ``n_features["target"]``.

    Notes
    -----
    Requires ``uni2ts>=2.0,<3`` (install separately on Python 3.10--3.12).
    Only target history is used; exogenous and categorical features are ignored.
    Fine-tuning is not supported: uni2ts generates non-differentiable samples.
    Forecasts are stochastic; use ``torch.manual_seed`` for reproducibility.
    """

    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.moirai._moirai_moe_pkg_v2 import (
            MoiraiMoE_pkg_v2,
        )

        return MoiraiMoE_pkg_v2

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-moe-1.0-R-small",
        module: nn.Module | None = None,
        num_samples: int = 100,
        patch_size: int = 16,
        metadata: dict | None = None,
    ):
        super().__init__(loss=MAE(), optimizer=None, metadata=metadata)
        for name, value in (
            ("context_length", self.context_length),
            ("prediction_length", self.prediction_length),
            ("target_dim", self.target_dim),
            ("num_samples", num_samples),
            ("patch_size", patch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        _check_soft_dependencies("uni2ts>=2.0,<3", obj=self)
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

        if module is None:
            module = MoiraiMoEModule.from_pretrained(model_id)
        if patch_size not in module.patch_sizes:
            raise ValueError(f"patch_size must be one of {module.patch_sizes}.")

        self.forecast = MoiraiMoEForecast(
            module=module,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            patch_size=patch_size,
            num_samples=num_samples,
        )
        self.forecast.requires_grad_(False)
        self.save_hyperparameters(ignore=["module"])

    @torch.no_grad()
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return means (B, H, D) and samples (B, S, H, D).

        ``history_target`` has shape (B, context_length, D), or (B, context_length)
        for a single target. ``history_mask`` is an optional (B, context_length)
        boolean mask with True for valid time steps. Non-finite targets are missing.
        Optional ``target_scale`` contains ``center`` and ``scale`` with shape
        (D,) or (B, D), for restoring the original units.
        """
        target = x["history_target"]
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        if target.ndim != 3 or target.shape[1:] != (
            self.context_length,
            self.target_dim,
        ):
            raise ValueError(
                "history_target must have shape "
                f"(batch, {self.context_length}, {self.target_dim})."
            )
        valid = x.get("history_mask")
        if valid is None:
            valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        elif valid.shape != target.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("history_mask must be boolean with shape (batch, time).")
        observed = torch.isfinite(target) & valid.unsqueeze(-1)
        self.forecast.eval()
        samples = self.forecast(
            past_target=target.masked_fill(~observed, 0),
            past_observed_target=observed,
            past_is_pad=~valid,
        )
        # uni2ts omits the target axis for univariate forecasts.
        if samples.ndim == 3:
            samples = samples.unsqueeze(-1)
        if "target_scale" in x:
            scale = x["target_scale"]["scale"].unsqueeze(-2).unsqueeze(-2)
            center = x["target_scale"]["center"].unsqueeze(-2).unsqueeze(-2)
            samples = samples * scale + center
        return {"prediction": samples.mean(dim=1), "samples": samples}

    def to_prediction(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return sample means, omitting the singleton univariate target axis."""
        return out["prediction"].squeeze(-1)

    def to_quantiles(
        self,
        out: dict[str, torch.Tensor],
        quantiles: list[float] | None = None,
    ) -> torch.Tensor:
        """Return empirical quantiles of shape (B, H, Q) or (B, H, D, Q)."""
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        samples = out["samples"]
        levels = samples.new_tensor(quantiles)
        if (
            levels.ndim != 1
            or levels.numel() == 0
            or not ((levels >= 0) & (levels <= 1)).all()
        ):
            raise ValueError("quantiles must be a non-empty list of values in [0, 1].")
        result = torch.quantile(samples, levels, dim=1).movedim(0, -1)
        return result.squeeze(-2) if self.target_dim == 1 else result

    def training_step(self, *args, **kwargs):
        """Reject fine-tuning through the sampling-only forecast interface."""
        raise NotImplementedError("MoiraiMoE supports zero-shot prediction only.")

    def configure_optimizers(self):
        """No optimizer is needed for a frozen foundation model."""
        return None
