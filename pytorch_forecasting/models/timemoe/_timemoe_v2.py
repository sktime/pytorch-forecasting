"""
Time Series Mixture of Experts (TimeMoE)
-----------------------------------------
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class TimeMoE(TslibBaseModel):
    """
    An implementation of the TimeMoE model for v2 of pytorch-forecasting.

    TimeMoE is a large pre-trained time series foundation model based on a
    decoder-only transformer architecture with sparse mixture-of-experts (MoE)
    feed-forward layers. It supports zero-shot and fine-tuned forecasting by
    loading weights from the ``Maple728/TimeMoE-50M`` (or ``-200M``) checkpoint
    on HuggingFace and generating future tokens autoregressively.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use for training / fine-tuning.
    pretrained_model_name : str, default='Maple728/TimeMoE-50M'
        HuggingFace model identifier for the pretrained TimeMoE checkpoint.
        Common options are ``'Maple728/TimeMoE-50M'`` and
        ``'Maple728/TimeMoE-200M'``.
    task_name : str, default='zero_shot_forecast'
        Forecasting task type.  Currently only ``'zero_shot_forecast'`` is
        supported; the ``forward`` pass will return ``None`` for any other
        value, matching the reference implementation behaviour.
    training : bool, default=True
        If ``False``, indicates the model is being used for evaluation-only
        (keeps original semantics of the replaced flag). When ``True``, the
        backbone parameters are left trainable for fine-tuning.
    logging_metrics : list[nn.Module] or None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer or str or None, default='adam'
        Optimizer to use for training. Can be a string name or an instance.
    optimizer_params : dict or None, default=None
        Parameters for the optimizer.
    lr_scheduler : str or None, default=None
        Learning rate scheduler name.  If ``None``, no scheduler is used.
    lr_scheduler_params : dict or None, default=None
        Parameters for the learning rate scheduler.
    metadata : dict or None, default=None
        Metadata for the model from ``TslibDataModule``.  Used to infer
        ``context_length``, ``prediction_length``, and ``target_dim``.

    References
    ----------
    [1] Shi, X. et al. (2024). Time-MoE: Billion-Scale Time Series Foundation
        Models with Mixture of Experts. https://arxiv.org/abs/2409.16040
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py
    [3] https://huggingface.co/Maple728/TimeMoE-50M

    Notes
    -----
    [1] This model requires the ``transformers`` library (``pip install
        "transformers<=4.40.1"``)
    """

    @classmethod
    def _pkg(cls):
        """Package containing the TimeMoE model."""
        from pytorch_forecasting.models.timemoe._timemoe_pkg_v2 import TimeMoE_pkg_v2

        return TimeMoE_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        pretrained_model_name: str = "Maple728/TimeMoE-50M",
        task_name: str = "zero_shot_forecast",
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

        self.pretrained_model_name = pretrained_model_name
        self.task_name = task_name
        self.training = training

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialize the TimeMoE network by loading the pretrained backbone.

        Loads the ``AutoModelForCausalLM`` checkpoint specified by
        ``pretrained_model_name`` from HuggingFace Hub (``trust_remote_code=True``
        is required for the TimeMoE custom modelling code).

        Raises
        ------
        ImportError
            If the ``transformers`` package is not installed.
        OSError
            If the model checkpoint cannot be loaded from HuggingFace Hub.
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "The `transformers` package is required to use TimeMoE. "
                "Install it with: pip install transformers"
            ) from e

        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name,
            trust_remote_code=True,
        )

        if not self.training:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Core forecasting logic following the TimeMoE reference implementation.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary.  Expected keys:

            - ``'history_cont'`` : ``(B, context_length, C)`` – historical
              continuous features used as encoder context.

        Returns
        -------
        torch.Tensor
            Denormalized forecast of shape ``(B, prediction_length, C)``.
        """
        x_enc = x["history_cont"]
        means = x_enc.mean(dim=1, keepdim=True).detach()  # (B, 1, C)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # (B, 1, C)
        x_enc = (x_enc - means) / stdev  # (B, L, C)

        B, L, C = x_enc.shape

        x_enc_flat = x_enc.permute(0, 2, 1).reshape(B * C, L)  # (B*C, L)

        if not self.training:  # just evaluation
            output = self.backbone.generate(
                x_enc_flat,
                max_new_tokens=self.prediction_length,
            )

            future_flat = output[:, -self.prediction_length :]

        else:
            outputs = self.backbone(x_enc_flat)
            logits = outputs.logits
            future_flat = logits[:, -self.prediction_length :, 0]

        dec_out = future_flat.reshape(B, C, self.prediction_length).permute(0, 2, 1)

        dec_out = dec_out * stdev + means  # broadcast (B, 1, C) → (B, pred_len, C)

        return dec_out

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimeMoE model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary.  See :meth:`_forecast` for the expected keys.
            If the key ``'target_scale'`` is present its value is forwarded to
            :meth:`transform_output` for inverse scaling after forecasting.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{'prediction': tensor}`` where ``tensor`` has shape
            ``(B, prediction_length, target_dim)``, or ``{'prediction': None}``
            if the ``task_name`` is not ``'zero_shot_forecast'``.
        """
        if self.task_name != "zero_shot_forecast":
            return {"prediction": None}

        prediction = self._forecast(x)  # (B, prediction_length, C)

        prediction = prediction[:, : self.prediction_length, : self.target_dim]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
