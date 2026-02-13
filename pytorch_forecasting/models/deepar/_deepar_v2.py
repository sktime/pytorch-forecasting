########################################################################################
# Disclaimer: This implementation is based on the new version of data pipeline and is
# experimental, please use with care.
########################################################################################

from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import (
    DistributionLoss,
    MultiLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.nn import HiddenState, get_rnn
from pytorch_forecasting.utils import apply_to_list


class DeepAR(BaseModel):
    """
    DeepAR: Probabilistic forecasting with autoregressive recurrent networks.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use.
    logging_metrics : list[nn.Module], optional
        Metrics to log during training.
    optimizer : Union[Optimizer, str], optional
        Optimizer to use. Defaults to "adam".
    optimizer_params : dict, optional
        Parameters for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler.
    lr_scheduler_params : dict, optional
        Parameters for the learning rate scheduler.
    cell_type : Literal["LSTM", "GRU"], optional
        Recurrent cell type ["LSTM", "GRU"]. Defaults to "LSTM".
    hidden_size : int, optional
        Hidden recurrent size. Defaults to 10.
    rnn_layers : int, optional
        Number of RNN layers. Defaults to 2.
    dropout : float, optional
        Dropout in RNN layers. Defaults to 0.1.
    metadata : dict, optional
        Metadata from the DataModule.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.deepar.__deepar_pkg_v2 import (
            DeepAR_pkg_v2,
        )

        return DeepAR_pkg_v2

    def __init__(
        self,
        loss: nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        cell_type: Literal["LSTM", "GRU"] = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        metadata: dict | None = None,
        output_transformer: Any = None,
        **kwargs: Any,
    ):
        if loss is None:
            loss = NormalDistributionLoss()
        if not isinstance(loss, (DistributionLoss, MultiLoss)):
            raise ValueError(
                f"DeepAR requires a 'DistributionLoss', but got {type(loss).__name__}. "
                "SMAPE is not supported as the primary training loss for DeepAR."
            )

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.metadata = metadata or {}
        self.output_transformer = output_transformer

        self.max_encoder_length = self.metadata.get("max_encoder_length", 0)
        self.max_prediction_length = self.metadata.get("max_prediction_length", 0)

        self.encoder_cont_dim = self.metadata.get("encoder_cont", 0)
        self.encoder_cat_dim = self.metadata.get("encoder_cat", 0)
        self.decoder_cont_dim = self.metadata.get("decoder_cont", 0)
        self.decoder_cat_dim = self.metadata.get("decoder_cat", 0)

        self.target_dim = self.metadata.get("target_dim", 1)

        rnn_class = get_rnn(cell_type)
        encoder_input_size = self.encoder_cont_dim + self.encoder_cat_dim
        decoder_input_size = self.decoder_cont_dim + self.decoder_cat_dim
        rnn_input_size = hidden_size

        self.encoder_projector = nn.Linear(encoder_input_size, rnn_input_size)
        self.decoder_projector = nn.Linear(decoder_input_size, rnn_input_size)

        self.rnn = rnn_class(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True,
        )
        if isinstance(self.loss, MultiLoss):
            n_outputs = sum(len(l.distribution_arguments) for l in self.loss)
        else:
            n_outputs = len(self.loss.distribution_arguments) * self.target_dim

        self.distribution_projector = nn.Linear(hidden_size, n_outputs)

        self.target_positions = torch.tensor([0])
        self.lagged_target_positions = {}
        self.n_reals = self.encoder_cont_dim
        self.n_categoricals = self.encoder_cat_dim

    @property
    def output_transformer(self):
        if hasattr(self, "_output_transformer"):
            return self._output_transformer
        return None

    @output_transformer.setter
    def output_transformer(self, value):
        self._output_transformer = value

    def on_fit_start(self):
        """Set output transformer from datamodule if available."""
        if self.output_transformer is None:
            if (
                hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
            ):
                if hasattr(self.trainer.datamodule, "target_normalizer"):
                    self.output_transformer = self.trainer.datamodule.target_normalizer

        if self.output_transformer is not None:
            self.hparams.output_transformer = self.output_transformer

    def transform_output(
        self,
        prediction: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        class DummyEncoder:
            transformation = None
            center = False

            @property
            def transform(self):
                return self.transformation

            def __call__(self, x):
                return x.get("prediction", 0)

        encoder = self.output_transformer
        if encoder is None or isinstance(encoder, str):
            encoder = DummyEncoder()

        return self.loss.rescale_parameters(
            prediction, target_scale=target_scale, encoder=encoder
        )

    def construct_input_vector(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        one_off_target: torch.Tensor = None,
        is_encoder: bool = True,
    ) -> torch.Tensor:
        """Create input vector into RNN network"""

        if self.n_reals > 0 and self.n_categoricals > 0:
            input_vector = torch.cat([x_cont, x_cat], dim=-1)
        elif self.n_reals > 0:
            input_vector = x_cont.clone()
        elif self.n_categoricals > 0:
            input_vector = x_cat.clone()
        else:
            raise ValueError("No features found in input")

        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )

        if one_off_target is not None:
            input_vector[:, 0, self.target_positions] = one_off_target.reshape(
                input_vector.size(0), -1
            )
        else:
            input_vector = input_vector[:, 1:]

        if is_encoder:
            input_vector = self.encoder_projector(input_vector)
        else:
            input_vector = self.decoder_projector(input_vector)

        return input_vector

    def encode(self, x: dict[str, torch.Tensor]) -> HiddenState:
        """Encode sequence into hidden state."""
        input_vector = self.construct_input_vector(
            x["encoder_cat"], x["encoder_cont"], is_encoder=True
        )
        _, hidden_state = self.rnn(input_vector)
        return hidden_state

    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(x, hidden_state)
        output = self.distribution_projector(decoder_output)
        return output, hidden_state

    def output_to_prediction(
        self,
        prediction_params: torch.Tensor,
        target_scale: torch.Tensor,
        n_samples: int = 1,
    ):
        """Convert network output to prediction and sample next target."""
        rescaled_params = self.transform_output(
            prediction_params, target_scale=target_scale
        )

        if n_samples > 1:
            prediction = self.loss.sample(rescaled_params, n_samples)
        else:
            prediction = self.loss.sample(rescaled_params, 1)

        if target_scale.ndim == 2 and target_scale.shape[-1] == 2:
            center = target_scale[..., 0].unsqueeze(-1)
            scale = target_scale[..., 1].unsqueeze(-1)
        else:
            if target_scale.shape[-1] == 2:
                center = target_scale[..., 0].unsqueeze(-1)
                scale = target_scale[..., 1].unsqueeze(-1)
            else:
                center = torch.zeros_like(target_scale).unsqueeze(-1)
                scale = target_scale.unsqueeze(-1)

        while scale.ndim < prediction.ndim:
            scale = scale.unsqueeze(-1)
            center = center.unsqueeze(-1)

        normalized_prediction = (prediction - center) / scale
        return prediction, normalized_prediction

    def decode_autoregressive(
        self,
        decode_one: callable,
        first_target: torch.Tensor,
        first_hidden_state: Any,
        target_scale: torch.Tensor,
        n_decoder_steps: int,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Make predictions in auto-regressive manner."""
        output = []
        current_hidden_state = first_hidden_state
        normalized_output = [first_target.unsqueeze(1)]

        for idx in range(n_decoder_steps):
            prediction_params, current_hidden_state = decode_one(
                idx,
                lagged_targets=normalized_output,
                hidden_state=current_hidden_state,
            )
            rescaled, normalized = self.output_to_prediction(
                prediction_params, target_scale, n_samples=n_samples
            )

            normalized_output.append(normalized.unsqueeze(1))
            output.append(rescaled)

        return torch.stack(output, dim=1)

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> torch.Tensor:
        """Decode hidden state into prediction."""
        if n_samples is None:
            output, _ = self.decode_all(input_vector, hidden_state)
            output = self.transform_output(output, target_scale)
            return output
        else:
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                lag_val = lagged_targets[-1].squeeze(1)
                x[:, 0, target_pos] = lag_val
                prediction_params, hidden_state = self.decode_all(x, hidden_state)
                return prediction_params[:, 0], hidden_state

            return self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
                n_samples=1,
            )

    def forward(
        self, x: dict[str, torch.Tensor], n_samples: int = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass using V1 logic."""
        target_scale = x.get(
            "target_scale",
            torch.ones((x["encoder_cont"].size(0), 1), device=x["encoder_cont"].device),
        )
        if target_scale.ndim == 1:
            target_scale = target_scale.unsqueeze(-1)
        if target_scale.shape[-1] == 1:
            target_scale = torch.cat(
                [torch.zeros_like(target_scale), target_scale], dim=-1
            )

        hidden_state = self.encode(x)
        target_pos = self.target_positions

        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0)),
            x["encoder_lengths"] - 1,
            target_pos,
        ]

        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=last_encoder_target,
            is_encoder=False,
        )

        if self.training:
            assert n_samples is None

        if n_samples is not None and n_samples > 1:
            batch_size = input_vector.size(0)
            input_vector = input_vector.repeat_interleave(n_samples, dim=0)
            hidden_state = apply_to_list(
                hidden_state, lambda t: t.repeat_interleave(n_samples, dim=0)
            )
            target_scale = target_scale.repeat_interleave(n_samples, dim=0)
            decode_samples = 1
        else:
            decode_samples = n_samples
            batch_size = input_vector.size(0)

        output = self.decode(
            input_vector,
            target_scale=target_scale,
            decoder_lengths=x["decoder_lengths"],
            hidden_state=hidden_state,
            n_samples=decode_samples,
        )

        if n_samples is not None and n_samples > 1:
            if output.ndim == 2:
                output = output.view(batch_size, n_samples, -1).permute(0, 2, 1)
            elif output.ndim == 3:
                output = output.view(batch_size, n_samples, output.size(1), -1).permute(
                    0, 2, 1, 3
                )
                if output.shape[-1] == 1:
                    output = output.squeeze(-1)

        return {"prediction": output}
