"""
Minimal Lightning model extension template for PyTorch Forecasting (v1).

PURPOSE:
--------
This is NOT a working model.
It is a structured template to help contributors implement new v1 models
that integrate cleanly with PyTorch Forecasting’s testing and API.

HOW TO USE:
-----------
- Copy this file and modify the class name.
- Implement the required methods marked below.
- Follow the comments carefully — they explain what each method should do.
"""

from typing import Any

import torch

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.models.base import BaseModel


class ExampleNetwork(BaseModel):
    """
    Minimal template model for contributors.

    TODO for contributors:
    -----------------------
    - Add your model hyperparameters to __init__
    - Implement forward()
    - Implement from_dataset()
    - Implement any additional methods required for your architecture
    """

    def __init__(self, hidden_size: int = 16, **kwargs):
        """
        Constructor for your model.

        REQUIRED:
        - Call super().__init__(**kwargs)

        TYPICAL PATTERN:
        - Save hyperparameters with self.save_hyperparameters()
        - Initialize layers (e.g., RNN, Transformer, MLP, etc.)
        """

        self.save_hyperparameters()
        super().__init__(**kwargs)

        # TODO: define your network layers here, e.g.:
        # self.rnn = torch.nn.LSTM(...)
        # self.projection = torch.nn.Linear(...)

    @classmethod
    def _pkg(cls):
        """
        REQUIRED for v1 models.

        Returns the package container class that defines metadata (_tags)
        and test fixtures.
        """
        from .package import ExampleNetwork_pkg

        return ExampleNetwork_pkg

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] | None = None,
        **kwargs,
    ):
        """
        REQUIRED factory method to construct model from a TimeSeriesDataSet.

        What you should do here:
        ------------------------
        - Extract needed information from `dataset`
        - Possibly modify kwargs (e.g., set loss, logging metrics, etc.)
        - Then call super().from_dataset()

        This ensures your model is correctly initialized from data.

        Example (simplified pattern used by most v1 models):

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **kwargs,
        )
        """
        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **kwargs,
        )

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        REQUIRED: implement the forward pass of your network.

        INPUT:
        ------
        x is a dictionary from TimeSeriesDataSet containing tensors such as:
        - x["encoder_cont"], x["encoder_cat"]
        - x["decoder_cont"], x["decoder_cat"]
        - x["encoder_lengths"], x["decoder_lengths"]
        - x["target_scale"], etc.

        WHAT YOU SHOULD DO HERE:
        ------------------------
        1) Encode past sequence (optional but common)
        2) Decode future sequence
        3) Produce predictions

        OUTPUT:
        -------
        You MUST return a dictionary created via:
            return self.to_network_output(prediction=your_prediction_tensor)

        The shape of prediction should typically be:
        (batch_size, decoder_length, target_dim)
        """

        # TODO: replace with your actual model logic
        raise NotImplementedError("Implement forward() in your custom model")

    def to_prediction(
        self, out: dict[str, Any], use_metric: bool = True, **kwargs
    ) -> torch.Tensor:
        """
        REQUIRED: convert raw network output to final prediction.

        - Called during inference.
        - Typically just returns out.prediction.
        """
        return out.prediction

    def to_quantiles(
        self, out: dict[str, Any], use_metric: bool = True, **kwargs
    ) -> torch.Tensor:
        """
        OPTIONAL: implement only if your model supports probabilistic outputs.

        If you do NOT support quantiles, you can delete this method.
        """
        # Example placeholder:
        return out.prediction[..., None]
