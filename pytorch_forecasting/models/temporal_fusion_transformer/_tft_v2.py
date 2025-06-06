########################################################################################
# Disclaimer: This implementation is based on the new version of data pipeline and is
# experimental, please use with care.
########################################################################################

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class TFT(BaseModel):
    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        metadata: Optional[dict] = None,
        output_size: int = 1,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.metadata = metadata
        self.output_size = output_size

        self.max_encoder_length = self.metadata["max_encoder_length"]
        self.max_prediction_length = self.metadata["max_prediction_length"]
        self.encoder_cont = self.metadata["encoder_cont"]
        self.encoder_cat = self.metadata["encoder_cat"]
        self.encoder_input_dim = self.encoder_cont + self.encoder_cat
        self.decoder_cont = self.metadata["decoder_cont"]
        self.decoder_cat = self.metadata["decoder_cat"]
        self.decoder_input_dim = self.decoder_cont + self.decoder_cat
        self.static_cat_dim = self.metadata.get("static_categorical_features", 0)
        self.static_cont_dim = self.metadata.get("static_continuous_features", 0)
        self.static_input_dim = self.static_cat_dim + self.static_cont_dim

        if self.encoder_input_dim > 0:
            self.encoder_var_selection = nn.Sequential(
                nn.Linear(self.encoder_input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.encoder_input_dim),
                nn.Sigmoid(),
            )
        else:
            self.encoder_var_selection = None

        if self.decoder_input_dim > 0:
            self.decoder_var_selection = nn.Sequential(
                nn.Linear(self.decoder_input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.decoder_input_dim),
                nn.Sigmoid(),
            )
        else:
            self.decoder_var_selection = None

        if self.static_input_dim > 0:
            self.static_context_linear = nn.Linear(self.static_input_dim, hidden_size)
        else:
            self.static_context_linear = None

        _lstm_encoder_input_actual_dim = self.encoder_input_dim
        self.lstm_encoder = nn.LSTM(
            input_size=max(1, _lstm_encoder_input_actual_dim),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        _lstm_decoder_input_actual_dim = self.decoder_input_dim
        self.lstm_decoder = nn.LSTM(
            input_size=max(1, _lstm_decoder_input_actual_dim),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_head_size,
            dropout=dropout,
            batch_first=True,
        )

        self.pre_output = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TFT model.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary containing input tensors:
            - encoder_cat: Categorical encoder features
            - encoder_cont: Continuous encoder features
            - decoder_cat: Categorical decoder features
            - decoder_cont: Continuous decoder features
            - static_categorical_features: Static categorical features
            - static_continuous_features: Static continuous features

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing output tensors:
            - prediction: Prediction output (batch_size, prediction_length, output_size)
        """
        batch_size = x["encoder_cont"].shape[0]

        encoder_cat = x.get(
            "encoder_cat",
            torch.zeros(batch_size, self.max_encoder_length, 0, device=self.device),
        )
        encoder_cont = x.get(
            "encoder_cont",
            torch.zeros(batch_size, self.max_encoder_length, 0, device=self.device),
        )
        decoder_cat = x.get(
            "decoder_cat",
            torch.zeros(batch_size, self.max_prediction_length, 0, device=self.device),
        )
        decoder_cont = x.get(
            "decoder_cont",
            torch.zeros(batch_size, self.max_prediction_length, 0, device=self.device),
        )

        encoder_input = torch.cat([encoder_cont, encoder_cat], dim=2)
        decoder_input = torch.cat([decoder_cont, decoder_cat], dim=2)

        static_context = None
        if self.static_context_linear is not None:
            static_cat = x.get(
                "static_categorical_features",
                torch.zeros(batch_size, 1, 0, device=self.device),
            )
            static_cont = x.get(
                "static_continuous_features",
                torch.zeros(batch_size, 1, 0, device=self.device),
            )

            if static_cat.size(2) == 0 and static_cont.size(2) == 0:
                static_context = None
            elif static_cat.size(2) == 0:
                static_input = static_cont.to(
                    dtype=self.static_context_linear.weight.dtype
                )
                static_context = self.static_context_linear(static_input)
                static_context = static_context.view(batch_size, self.hidden_size)
            elif static_cont.size(2) == 0:
                static_input = static_cat.to(
                    dtype=self.static_context_linear.weight.dtype
                )
                static_context = self.static_context_linear(static_input)
                static_context = static_context.view(batch_size, self.hidden_size)
            else:
                static_input = torch.cat([static_cont, static_cat], dim=2).to(
                    dtype=self.static_context_linear.weight.dtype
                )
                static_context = self.static_context_linear(static_input)
                static_context = static_context.view(batch_size, self.hidden_size)

        if self.encoder_var_selection is not None:
            encoder_weights = self.encoder_var_selection(encoder_input)
            encoder_input = encoder_input * encoder_weights
        else:
            if self.encoder_input_dim == 0:
                encoder_input = torch.zeros(
                    batch_size,
                    self.max_encoder_length,
                    1,
                    device=self.device,
                    dtype=encoder_input.dtype,
                )
            else:
                encoder_input = encoder_input

        if self.decoder_var_selection is not None:
            decoder_weights = self.decoder_var_selection(decoder_input)
            decoder_input = decoder_input * decoder_weights
        else:
            if self.decoder_input_dim == 0:
                decoder_input = torch.zeros(
                    batch_size,
                    self.max_prediction_length,
                    1,
                    device=self.device,
                    dtype=decoder_input.dtype,
                )
            else:
                decoder_input = decoder_input

        if static_context is not None:
            encoder_static_context = static_context.unsqueeze(1).expand(
                -1, self.max_encoder_length, -1
            )
            decoder_static_context = static_context.unsqueeze(1).expand(
                -1, self.max_prediction_length, -1
            )

            encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_input)
            encoder_output = encoder_output + encoder_static_context
            decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))
            decoder_output = decoder_output + decoder_static_context
        else:
            encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_input)
            decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))

        sequence = torch.cat([encoder_output, decoder_output], dim=1)

        if static_context is not None:
            expanded_static_context = static_context.unsqueeze(1).expand(
                -1, sequence.size(1), -1
            )

            attended_output, _ = self.self_attention(
                sequence + expanded_static_context, sequence, sequence
            )
        else:
            attended_output, _ = self.self_attention(sequence, sequence, sequence)

        decoder_attended = attended_output[:, -self.max_prediction_length :, :]

        output = nn.functional.relu(self.pre_output(decoder_attended))
        prediction = self.output_layer(output)

        return {"prediction": prediction}
