########################################################################################
# Disclaimer: This implementation is based on the new version of data pipeline and is
# experimental, please use with care.
########################################################################################

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base.base_model_refactor import BaseModel


class TFT(BaseModel):
    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: Optional[List[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[Dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Dict] = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        metadata: Optional[Dict] = None,
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
        self.static_categorical_features = self.metadata["static_categorical_features"]
        self.static_continuous_features = self.metadata["static_continuous_features"]

        total_feature_size = self.encoder_cont + self.encoder_cat
        total_static_size = (
            self.static_categorical_features + self.static_continuous_features
        )

        self.encoder_var_selection = nn.Sequential(
            nn.Linear(total_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, total_feature_size),
            nn.Sigmoid(),
        )

        self.decoder_var_selection = nn.Sequential(
            nn.Linear(total_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, total_feature_size),
            nn.Sigmoid(),
        )

        self.static_context_linear = (
            nn.Linear(total_static_size, hidden_size) if total_static_size > 0 else None
        )

        self.lstm_encoder = nn.LSTM(
            input_size=total_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=total_feature_size,
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
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
                torch.zeros(batch_size, 0, device=self.device),
            )
            static_cont = x.get(
                "static_continuous_features",
                torch.zeros(batch_size, 0, device=self.device),
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

                static_input = torch.cat([static_cont, static_cat], dim=1).to(
                    dtype=self.static_context_linear.weight.dtype
                )
                static_context = self.static_context_linear(static_input)
                static_context = static_context.view(batch_size, self.hidden_size)

        encoder_weights = self.encoder_var_selection(encoder_input)
        encoder_input = encoder_input * encoder_weights

        decoder_weights = self.decoder_var_selection(decoder_input)
        decoder_input = decoder_input * decoder_weights

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
