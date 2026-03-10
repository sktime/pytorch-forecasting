import torch
import torch.nn as nn

from pytorch_forecasting.layers import RevIN
from pytorch_forecasting.models.autoformer.layers import Decoder, Encoder
from pytorch_forecasting.models.base._base_model_v2 import BaseModelV2


class Autoformer(BaseModelV2):
    """Autoformer is a transformer-based architecture designed specifically
    for long-horizon time series forecasting. Unlike standard transformers,
    it replaces self-attention with an Auto-Correlation mechanism and
    incorporates progressive series decomposition into seasonal and trend
    components.

    Parameters
    ----------
    d_model : int, optional
        Hidden dimension of the model used for encoder and decoder
        representations. Default is 32.
    enc_layers : int, optional
        Number of encoder layers. Default is 1.
    dec_layers : int, optional
        Number of decoder layers. Default is 1.
    moving_avg : int, optional
        Window size used in the series decomposition moving-average
        filter. Default is 25.
    use_revin : bool, optional
        Whether to apply Reversible Instance Normalization (RevIN)
        to stabilize training under distribution shifts. Default is False.

    """

    def __init__(
        self,
        d_model: int = 32,
        enc_layers: int = 1,
        dec_layers: int = 1,
        moving_avg: int = 25,
        use_revin: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(self.input_size)

        self.input_proj = nn.Linear(self.input_size, d_model)

        self.encoder = Encoder(
            d_model=d_model,
            num_layers=enc_layers,
            moving_avg=moving_avg,
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_layers=dec_layers,
            moving_avg=moving_avg,
        )

        self.seasonal_proj = nn.Linear(d_model, self.output_size)
        self.trend_proj = nn.Linear(d_model, self.output_size)

    def forward(self, x):
        """
        Forward pass of the Autoformer model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input dictionary containing model inputs produced by the
            EncoderDecoderTimeSeriesDataModule.

            Expected keys include:

            * ``encoder_cont`` : torch.Tensor
              Continuous encoder inputs with shape
              (batch_size, encoder_length, input_size).

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs containing:

            * ``prediction`` : torch.Tensor
              Forecast tensor of shape
              (batch_size, prediction_length, output_size).
        """
        enc_input = x["encoder_cont"]

        if self.use_revin:
            enc_input = self.revin(enc_input, mode="norm")

        enc_input = self.input_proj(enc_input)
        enc_out, trend = self.encoder(enc_input)

        dec_out, dec_trend = self.decoder(enc_out, enc_out)

        seasonal = self.seasonal_proj(dec_out)
        trend = self.trend_proj(dec_trend)

        output = seasonal + trend

        if self.use_revin:
            output = self.revin(output, mode="denorm")

        return {"prediction": output}
