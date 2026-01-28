from copy import copy
from typing import Optional

import torch
from torch import nn

from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, Metric, MultiHorizonMetric, MultiLoss
from pytorch_forecasting.models.base import AutoRegressiveBaseModel
from pytorch_forecasting.models.nn import LSTM


class LSTMModel(AutoRegressiveBaseModel):
    """
    LSTM model for univariate and multivariate time series forecasting.

    This model uses an LSTM encoder-decoder architecture with autoregressive decoding.
    Supports both single-target and multi-target forecasting scenarios.

    Args:
        target: Target variable name. Can be a string for single target or
            list[str] for multi-target.
        target_lags: Dictionary of target names mapped to lag configurations.
            Required for autoregressive models but not used without covariates.
        hidden_size: Hidden size of the LSTM layers. Defaults to 10.
        n_layers: Number of LSTM layers. Defaults to 2.
        dropout: Dropout rate applied between LSTM layers. Defaults to 0.1.
        output_size: Output size per target. Can be int for single target or
            list[int] for multi-target. Defaults to 1.
        loss: Loss function. For multi-target, should be MultiLoss. Defaults to MAE().
        **kwargs: Additional arguments passed to BaseModel.
    """

    def __init__(
        self,
        target: str | list[str],
        target_lags: dict[str, dict[str, int]],
        hidden_size: int = 10,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_size: int | list[int] = 1,
        loss: MultiHorizonMetric = None,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__(loss=loss, **kwargs)

        # Determine number of targets and input/output sizes
        # Note: `n_targets` is a read-only property on BaseModel inferred from the loss.
        n_targets = self.n_targets

        # Handle multi-target output_size
        if isinstance(output_size, list):
            if len(output_size) != n_targets:
                raise ValueError(
                    f"output_size list length ({len(output_size)}) must match "
                    f"number of targets ({n_targets})"
                )
            self.output_sizes = output_size
        else:
            self.output_sizes = [output_size] * n_targets

        input_size = n_targets

        # LSTM network
        self.lstm = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size=input_size,  # Dynamic based on n_targets
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )

        # Output layers: separate head per target for multi-target
        if n_targets > 1:
            self.output_layers = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, size)
                    for size in self.output_sizes
                ]
            )
        else:
            self.output_layer = nn.Linear(
                self.hparams.hidden_size, self.output_sizes[0]
            )

    def encode(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence into hidden state.

        Args:
            x: Dictionary containing encoder_cont and encoder_lengths.

        Returns:
            Hidden state tuple (h_n, c_n) for LSTM.
        """
        assert x["encoder_lengths"].min() >= 1, "encoder_lengths must be >= 1"

        # Get target values from encoder_cont
        input_vector = x["encoder_cont"].clone()

        # Lag target by one time step for autoregressive prediction
        input_vector = torch.roll(input_vector, shifts=1, dims=1)

        # Remove first time step (cannot be used due to lagging)
        input_vector = input_vector[:, 1:]

        # Effective encoder lengths (reduced by 1 due to lagging)
        effective_encoder_lengths = x["encoder_lengths"] - 1

        # Run through LSTM
        _, hidden_state = self.lstm(
            input_vector,
            lengths=effective_encoder_lengths,
            enforce_sorted=False,
        )

        return hidden_state

    def decode(
        self,
        x: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Decode hidden state into predictions.

        Args:
            x: Dictionary containing decoder_cont, encoder_cont, encoder_lengths,
                decoder_lengths, and target_scale.
            hidden_state: Hidden state from encoder.

        Returns:
            Predictions tensor(s). For multi-target, returns list of tensors.
        """
        # Get decoder input
        input_vector = x["decoder_cont"].clone()

        # Lag target by one
        input_vector = torch.roll(input_vector, shifts=1, dims=1)

        # Fill first time step with last encoder target
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            :,  # All target dimensions
        ]
        input_vector[:, 0, :] = last_encoder_target

        if self.training:
            lstm_output, _ = self.lstm(
                input_vector,
                hidden_state,
                enforce_sorted=False,
            )

            # Apply output layers
            if self.n_targets > 1:
                predictions = [layer(lstm_output) for layer in self.output_layers]
            else:
                predictions = self.output_layer(lstm_output)

            # Transform output (denormalize)
            predictions = self.transform_output(
                predictions, target_scale=x["target_scale"]
            )

            return predictions

        else:
            # Inference mode: autoregressive decoding
            target_positions = torch.arange(self.n_targets, device=input_vector.device)

            def decode_one(idx, lagged_targets, hidden_state):
                """Decode one step ahead."""
                x_step = input_vector[:, [idx]]

                # Overwrite target positions with lagged targets
                if self.n_targets > 1:
                    # Multi-target: lagged_targets is list of tensors
                    for i, target_pos in enumerate(target_positions):
                        x_step[:, 0, target_pos] = lagged_targets[-1][:, i]
                else:
                    # Single target: lagged_targets is tensor
                    x_step[:, 0, target_positions[0]] = lagged_targets[-1]

                # Forward through LSTM
                lstm_output, hidden_state = self.lstm(x_step, hidden_state)
                lstm_output = lstm_output[:, 0]  # Take first timestep

                # Apply output layers
                if self.n_targets > 1:
                    output = [layer(lstm_output) for layer in self.output_layers]
                else:
                    output = self.output_layer(lstm_output)

                return output, hidden_state

            first_target = input_vector[:, 0, target_positions]
            if self.n_targets == 1:
                first_target = first_target.squeeze(-1)

            output = self.decode_autoregressive(
                decode_one,
                first_target=first_target,
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1),
            )
            # For multi-target, keep as list so MultiLoss
            # receives y_pred[idx] per target.
            # Do not concatenate here; forward() keeps list for n_targets > 1.
            return output

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input dictionary from dataloader.

        Returns:
            Dictionary with 'prediction' key containing predictions.
        """
        hidden_state = self.encode(x)
        output = self.decode(x, hidden_state)

        # (batch, time); do not concatenate to (batch, time, n_targets).
        if isinstance(output, list) and self.n_targets == 1:
            output = output[0]
        elif isinstance(output, list) and self.n_targets > 1:
            pass  # keep list for MultiLoss
        # else single tensor (inference branch already concatenated)

        return self.to_network_output(prediction=output)

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: TimeSeriesDataSet instance.
            **kwargs: Additional hyperparameters.

        Returns:
            LSTMModel instance.
        """
        # Validate dataset
        from pytorch_forecasting.data.encoders import MultiNormalizer

        assert not isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ), "Categorical targets not supported - target must be continuous"

        if isinstance(dataset.target_normalizer, MultiNormalizer):
            assert all(
                not isinstance(norm, NaNLabelEncoder)
                for norm in dataset.target_normalizer.normalizers
            ), "Categorical targets not supported"

        # Deduce output parameters (handles multi-target automatically)
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            cls.deduce_default_output_parameters(
                dataset=dataset, kwargs=kwargs, default_loss=MAE()
            )
        )

        return super().from_dataset(dataset, **new_kwargs)
