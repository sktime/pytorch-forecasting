"""DeepAR v2: Probabilistic forecasting with autoregressive recurrent networks.

This is a v2 implementation using the new BaseModel and DataModule interfaces.
Based on the paper:
`DeepAR: Probabilistic forecasting with autoregressive recurrent networks
<https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class DeepAR(BaseModel):
    """DeepAR v2: Probabilistic forecasting with autoregressive recurrent networks.

    This model uses an RNN-based encoder-decoder architecture with autoregressive
    decoding for probabilistic time series forecasting. During training, teacher
    forcing is used for efficient learning. During prediction, the model decodes
    autoregressively by feeding predicted distribution means back as input.

    The model outputs distribution parameters (e.g., loc and scale for a Normal
    distribution), and the loss function (e.g., ``NormalDistributionLoss``)
    computes the negative log-likelihood.

    Parameters
    ----------
    loss : nn.Module
        Distribution loss function (e.g., ``NormalDistributionLoss``).
        Must have ``distribution_arguments`` attribute and ``to_prediction``
        method.
    logging_metrics : list[nn.Module], optional
        Metrics to log during training. Defaults to None.
    optimizer : Optimizer or str, optional
        Optimizer configuration. Defaults to "adam".
    optimizer_params : dict, optional
        Parameters for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Learning rate scheduler parameters.
    metadata : dict, optional
        Metadata from DataModule containing dimension and length information.
        Expected keys: ``encoder_cont``, ``encoder_cat``, ``decoder_cont``,
        ``decoder_cat``, ``target``, ``max_encoder_length``,
        ``max_prediction_length``.
    cell_type : str, optional
        RNN cell type, either ``"LSTM"`` or ``"GRU"``. Defaults to ``"LSTM"``.
    hidden_size : int, optional
        Hidden size for RNN layers. Defaults to 32.
    rnn_layers : int, optional
        Number of stacked RNN layers. Defaults to 2.
    dropout : float, optional
        Dropout rate applied between RNN layers. Defaults to 0.1.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.deepar._deepar_pkg_v2 import DeepAR_pkg_v2

        return DeepAR_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        cell_type: str = "LSTM",
        hidden_size: int = 32,
        rnn_layers: int = 2,
        dropout: float = 0.1,
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

        self.metadata = metadata or {}
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.cell_type = cell_type
        self.transformation = None

        # Read dimensions from metadata
        self.max_encoder_length = self.metadata.get("max_encoder_length", 30)
        self.max_prediction_length = self.metadata.get("max_prediction_length", 1)
        self.encoder_cont_dim = self.metadata.get("encoder_cont", 0)
        self.encoder_cat_dim = self.metadata.get("encoder_cat", 0)
        self.decoder_cont_dim = self.metadata.get("decoder_cont", 0)
        self.decoder_cat_dim = self.metadata.get("decoder_cat", 0)
        self.n_targets = self.metadata.get("target", 1)

        # Distribution output dimension from loss function
        if hasattr(loss, "distribution_arguments"):
            self.n_dist_args = len(loss.distribution_arguments)
        else:
            self.n_dist_args = self.n_targets

        # Input dimensions:
        # Encoder sees all features + shifted target (autoregressive input)
        encoder_input_dim = (
            self.encoder_cont_dim + self.encoder_cat_dim + self.n_targets
        )
        # Decoder sees known future features + autoregressive target input
        decoder_input_dim = (
            self.decoder_cont_dim + self.decoder_cat_dim + self.n_targets
        )

        # Input projections to map different input dims to hidden_size
        self.encoder_input_proj = nn.Linear(max(1, encoder_input_dim), hidden_size)
        self.decoder_input_proj = nn.Linear(max(1, decoder_input_dim), hidden_size)

        # Shared RNN for encoder and decoder (preserves DeepAR architecture)
        rnn_cls = nn.LSTM if cell_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True,
        )

        # Projects RNN output to distribution parameters
        self.distribution_projector = nn.Linear(hidden_size, self.n_dist_args)

    def _build_encoder_input(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the encoder input by concatenating features and shifted target.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch dictionary.

        Returns
        -------
        torch.Tensor
            Encoder input of shape (batch, enc_len, encoder_input_dim).
        """
        target_past = x["target_past"]
        if target_past.dim() == 2:
            target_past = target_past.unsqueeze(-1)

        # Scale target past using target_scale
        target_scale = x["target_scale"]
        if target_scale.dim() == 1:
            target_scale_unsqueezed = target_scale.unsqueeze(1).unsqueeze(2)
        elif target_scale.dim() == 2 and target_scale.size(1) == 1:
            target_scale_unsqueezed = target_scale.unsqueeze(2)
        else:
            target_scale_unsqueezed = target_scale
        target_past_scaled = target_past / target_scale_unsqueezed

        # Shift target by 1 for autoregressive input: at time t, see target[t-1]
        shifted_target = torch.roll(target_past_scaled, shifts=1, dims=1)
        shifted_target[:, 0, :] = 0.0

        parts = []
        encoder_cont = x.get("encoder_cont")
        if encoder_cont is not None and encoder_cont.shape[-1] > 0:
            parts.append(encoder_cont)

        encoder_cat = x.get("encoder_cat")
        if encoder_cat is not None and encoder_cat.shape[-1] > 0:
            parts.append(encoder_cat.float())

        parts.append(shifted_target)

        return torch.cat(parts, dim=-1)

    def _build_decoder_input_teacher(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        last_target: torch.Tensor,
    ) -> torch.Tensor:
        """Build decoder input with teacher forcing.

        Uses ground-truth targets shifted by one step as autoregressive input.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch dictionary.
        y : torch.Tensor
            Ground-truth future targets of shape (batch, pred_len).
        last_target : torch.Tensor
            Last encoder target of shape (batch, 1, n_targets).

        Returns
        -------
        torch.Tensor
            Decoder input of shape (batch, pred_len, decoder_input_dim).
        """
        target_scale = x["target_scale"]
        if target_scale.dim() == 1:
            target_scale_unsqueezed = target_scale.unsqueeze(1).unsqueeze(2)
        elif target_scale.dim() == 2 and target_scale.size(1) == 1:
            target_scale_unsqueezed = target_scale.unsqueeze(2)
        else:
            target_scale_unsqueezed = target_scale

        # Scale last_target and y_input
        y_input = y.unsqueeze(-1) if y.dim() == 2 else y
        y_input_scaled = y_input / target_scale_unsqueezed
        last_target_scaled = last_target / target_scale_unsqueezed

        # Prepare shifted target: [last_encoder_target, y[0], y[1], ..., y[T-2]]
        teacher_target = torch.cat(
            [last_target_scaled, y_input_scaled[:, :-1, :]], dim=1
        )

        parts = []
        decoder_cont = x.get("decoder_cont")
        if decoder_cont is not None and decoder_cont.shape[-1] > 0:
            parts.append(decoder_cont)

        decoder_cat = x.get("decoder_cat")
        if decoder_cat is not None and decoder_cat.shape[-1] > 0:
            parts.append(decoder_cat.float())

        parts.append(teacher_target)

        return torch.cat(parts, dim=-1)

    def _decode_autoregressive(
        self,
        x: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, ...] | torch.Tensor,
        last_target: torch.Tensor,
    ) -> torch.Tensor:
        """Decode autoregressively using predicted means as next-step input.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch dictionary.
        hidden_state : tuple or torch.Tensor
            RNN hidden state from encoder.
        last_target : torch.Tensor
            Last encoder target of shape (batch, 1, n_targets).

        Returns
        -------
        torch.Tensor
            Distribution parameters of shape (batch, pred_len, n_dist_args).
        """
        # Prepare target scale for internal scaling and division
        target_scale_val = x["target_scale"]
        if target_scale_val.dim() == 1:
            target_scale_expanded = target_scale_val.unsqueeze(1).unsqueeze(2)
        elif target_scale_val.dim() == 2 and target_scale_val.size(1) == 1:
            target_scale_expanded = target_scale_val.unsqueeze(2)
        else:
            target_scale_expanded = target_scale_val

        # Scale last target
        current_target = last_target / target_scale_expanded

        # Prepare known future features
        known_parts = []
        decoder_cont = x.get("decoder_cont")
        if decoder_cont is not None and decoder_cont.shape[-1] > 0:
            known_parts.append(decoder_cont)

        decoder_cat = x.get("decoder_cat")
        if decoder_cat is not None and decoder_cat.shape[-1] > 0:
            known_parts.append(decoder_cat.float())

        if known_parts:
            known_features = torch.cat(known_parts, dim=-1)
            has_known = True
        else:
            has_known = False

        predictions = []

        for t in range(self.max_prediction_length):
            step_parts = []
            if has_known:
                step_parts.append(known_features[:, t : t + 1, :])
            step_parts.append(current_target)

            step_input = torch.cat(step_parts, dim=-1)
            step_input = self.decoder_input_proj(step_input)

            step_output, hidden_state = self.rnn(step_input, hidden_state)
            step_params = self.distribution_projector(step_output)
            predictions.append(step_params)

            # Rescale step_params for the loss functions to map to distribution
            rescaled_step_params = self.rescale_prediction(
                step_params, target_scale_val
            )

            # Use distribution mean as next autoregressive input
            if hasattr(self.loss, "to_prediction"):
                try:
                    predicted_mean = self.loss.to_prediction(rescaled_step_params)
                except Exception:
                    predicted_mean = rescaled_step_params[..., 2]
            else:
                predicted_mean = rescaled_step_params[..., 2]

            # predicted_mean is unscaled, so divide by target_scale for next step
            if predicted_mean.dim() == 2:
                current_target = predicted_mean.unsqueeze(-1) / target_scale_expanded
            else:
                current_target = predicted_mean / target_scale_expanded

        return torch.cat(predictions, dim=1)

    def rescale_prediction(
        self, prediction: torch.Tensor, target_scale: torch.Tensor
    ) -> torch.Tensor:
        """Rescale prediction to original/real space.

        For DistributionLoss, this maps the 2-parameter output to the 4-parameter
        tensor expected by the loss.
        """
        from pytorch_forecasting.metrics.base_metrics import DistributionLoss

        if isinstance(self.loss, DistributionLoss):
            if self.loss.__class__.__name__ == "LogNormalDistributionLoss":
                self.transformation = "log"
            else:
                self.transformation = None

            if (
                not hasattr(self.loss, "_transformation")
                or self.loss._transformation is None
            ):
                self.loss._transformation = self.transformation

            if target_scale.dim() == 1:
                target_scale = torch.stack(
                    [torch.zeros_like(target_scale), target_scale], dim=-1
                )
            elif target_scale.dim() == 2 and target_scale.size(1) == 1:
                target_scale = torch.cat(
                    [torch.zeros_like(target_scale), target_scale], dim=-1
                )
            return self.loss.rescale_parameters(prediction, target_scale, encoder=self)
        else:
            if target_scale.dim() == 1:
                target_scale_unsqueezed = target_scale.unsqueeze(1).unsqueeze(2)
            elif target_scale.dim() == 2 and target_scale.size(1) == 1:
                target_scale_unsqueezed = target_scale.unsqueeze(2)
            else:
                target_scale_unsqueezed = target_scale
            return prediction * target_scale_unsqueezed

    def forward(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the DeepAR model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input dictionary from DataModule containing:
            - ``encoder_cont``: continuous encoder features
            - ``encoder_cat``: categorical encoder features
            - ``decoder_cont``: known continuous decoder features
            - ``decoder_cat``: known categorical decoder features
            - ``target_past``: historical target values
        y : torch.Tensor, optional
            Ground-truth future targets for teacher forcing during training.
            If None, autoregressive decoding is used.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with ``"prediction"`` key containing distribution
            parameters of shape ``(batch, pred_len, n_dist_args)``.
        """
        # === ENCODER ===
        encoder_input = self._build_encoder_input(x)
        encoder_input = self.encoder_input_proj(encoder_input)
        _, hidden_state = self.rnn(encoder_input)

        # Last encoder target as seed for decoder
        target_past = x["target_past"]
        if target_past.dim() == 2:
            target_past = target_past.unsqueeze(-1)
        last_target = target_past[:, -1:, :]

        # === DECODER ===
        if y is not None:
            # Teacher forcing: use ground-truth targets
            decoder_input = self._build_decoder_input_teacher(x, y, last_target)
            decoder_input = self.decoder_input_proj(decoder_input)
            decoder_output, _ = self.rnn(decoder_input, hidden_state)
            prediction = self.distribution_projector(decoder_output)
        else:
            # Autoregressive: use predicted means
            prediction = self._decode_autoregressive(x, hidden_state, last_target)

        # Rescale prediction
        prediction = self.rescale_prediction(prediction, x["target_scale"])

        return {"prediction": prediction}

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Training step with teacher forcing.

        Overrides base to pass ground-truth targets for teacher forcing.

        Parameters
        ----------
        batch : tuple
            Tuple of (x, y) where x is the input dict and y is targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the loss.
        """
        x, y = batch
        y_hat_dict = self(x, y=y)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_metrics(y_hat, y, prefix="train")
        return {"loss": loss}

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Validation step with teacher forcing.

        Overrides base to pass ground-truth targets for teacher forcing.

        Parameters
        ----------
        batch : tuple
            Tuple of (x, y) where x is the input dict and y is targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the validation loss.
        """
        x, y = batch
        y_hat_dict = self(x, y=y)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_metrics(y_hat, y, prefix="val")
        return {"val_loss": loss}

    def test_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Test step with teacher forcing.

        Overrides base to pass ground-truth targets for teacher forcing.

        Parameters
        ----------
        batch : tuple
            Tuple of (x, y) where x is the input dict and y is targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the test loss.
        """
        x, y = batch
        y_hat_dict = self(x, y=y)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_metrics(y_hat, y, prefix="test")
        return {"test_loss": loss}
