"""
Moirai Model for PyTorch Forecasting
------------------------------------
"""

from typing import Any, Optional, Union
from warnings import warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Moirai(TslibBaseModel):
    """
    Salesforce Moirai Model wrapper for PyTorch Forecasting.

    Parameters
    ----------
    module: MoiraiFinetune or MoiraiForecast or similar, optional
        The underlying Moirai module from uni2ts.
    loss : Descendants of ``pytorch_forecasting.metrics.Metric`` class
        Loss function to use for training.
    """

    def __init__(
        self,
        module: Any | None = None,
        loss: Metric | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
            **kwargs,
        )
        self.module = module

    @classmethod
    def _pkg(cls):
        """Return the package class for this model."""
        from pytorch_forecasting.models.moirai._moirai_pkg_v2 import Moirai_pkg_v2

        return Moirai_pkg_v2

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors (pytorch-forecasting format)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors
        """
        if self.module is not None:
            # Map PTF dataloader format to uni2ts format.
            # In PTF, x usually has:
            # - target_past (usually mapped to past_target)
            # - encoder_cont (often mapped to feat_dynamic_real)

            # This is a naive translation from the PTF representation to the inputs
            # expected by Moirai/uni2ts. It acts as an adapter layer so the
            # foundation model gets the tensors in the expected shape/types.
            kwargs = {}
            if "target_past" in x:
                kwargs["past_target"] = x["target_past"]

            if "encoder_cont" in x:
                kwargs["past_feat_dynamic_real"] = x["encoder_cont"]

            # past_observed_target and past_is_pad are required by uni2ts models
            # We can construct these proxies based on the past target if not provided
            if "past_target" in kwargs:
                # Assuming valid data is not NaN and padding is filled with 0s/NaNs
                # (which PTF handles with target_past_mask if available)
                if "target_past_mask" in x:
                    kwargs["past_observed_target"] = x["target_past_mask"].bool()
                else:
                    kwargs["past_observed_target"] = ~torch.isnan(kwargs["past_target"])

                # Padding detection
                if "encoder_lengths" in x:
                    # Construct padding mask from encoder_lengths
                    batch_size = kwargs["past_target"].size(0)
                    seq_len = kwargs["past_target"].size(1)
                    idx = torch.arange(seq_len, device=kwargs["past_target"].device)
                    idx = idx.unsqueeze(0).expand(batch_size, -1)
                    lengths = x["encoder_lengths"].unsqueeze(1)
                    kwargs["past_is_pad"] = idx >= lengths
                else:
                    kwargs["past_is_pad"] = torch.zeros(
                        kwargs["past_target"].shape[:2],
                        dtype=torch.bool,
                        device=kwargs["past_target"].device,
                    )

            # Perform prediction using uni2ts module
            pred = self.module(**kwargs)
            return {"prediction": pred}

        raise NotImplementedError("Underlying module is not provided.")
