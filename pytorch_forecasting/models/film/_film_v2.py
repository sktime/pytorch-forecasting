"""
FiLM model for Pytorch Forecasting.
------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of FiLM model for PTF v2.                     #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy import special as ss
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


def _hippo_legendre_transition(order: int):
    """Translated Legendre (LegT) HiPPO matrices.

    Parameters
    ----------
    order: int
        Order of the HiPPO projection.

    Returns
    -------
    tuple of np.ndarray
        The continuous-time ``(A, B)`` matrices of the LegT HiPPO operator,
        of shapes ``(order, order)`` and ``(order, 1)``.
    """
    poly_idx = np.arange(order, dtype=np.float64)
    poly_scale = (2 * poly_idx + 1)[:, None]
    col_idx, row_idx = np.meshgrid(poly_idx, poly_idx)
    leg_a = np.where(col_idx < row_idx, -1, (-1.0) ** (row_idx - col_idx + 1))
    leg_a = leg_a * poly_scale
    leg_b = (-1.0) ** poly_idx[:, None] * poly_scale
    return leg_a, leg_b


class HiPPOLegT(nn.Module):
    """HiPPO-LegT projection of a sequence onto Legendre polynomials.

    The module unrolls the discretized linear dynamics over the input
    sequence and returns the per-step Legendre coefficients.

    Parameters
    ----------
    order: int
        Order of the HiPPO projection (number of Legendre polynomials).
    dt: float, default=1.0
        Discretization step size. Should be roughly inverse to the length
        of the input sequence.
    discretization: str, default="bilinear"
        Discretization method passed to
        :func:`scipy.signal.cont2discrete`.
    """

    def __init__(
        self,
        order: int,
        dt: float = 1.0,
        discretization: str = "bilinear",
    ):
        super().__init__()
        self.order = order
        leg_a, leg_b = _hippo_legendre_transition(order)
        proj_c = np.ones((1, order))
        proj_d = np.zeros((1,))
        leg_a, leg_b, _, _, _ = signal.cont2discrete(
            (leg_a, leg_b, proj_c, proj_d), dt=dt, method=discretization
        )
        leg_b = leg_b.squeeze(-1)

        self.register_buffer("hippo_a", torch.as_tensor(leg_a, dtype=torch.float32))
        self.register_buffer("hippo_b", torch.as_tensor(leg_b, dtype=torch.float32))
        steps = np.arange(0.0, 1.0, dt)
        eval_matrix = ss.eval_legendre(np.arange(order)[:, None], 1 - 2 * steps).T
        self.register_buffer(
            "eval_matrix",
            torch.as_tensor(np.ascontiguousarray(eval_matrix), dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the input sequence onto Legendre polynomials.

        Parameters
        ----------
        inputs: torch.Tensor
            Input of shape ``(..., seq_len)`` where the last dimension
            is the time dimension.

        Returns
        -------
        torch.Tensor
            Legendre coefficients of shape ``(seq_len, ..., order)``.
        """
        hippo_a = self.hippo_a.to(dtype=inputs.dtype)
        hippo_b = self.hippo_b.to(dtype=inputs.dtype)
        state = torch.zeros(
            inputs.shape[:-1] + (self.order,),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        states = []
        for step in inputs.movedim(-1, 0):
            step = step.unsqueeze(-1)
            state = F.linear(state, hippo_a) + step @ hippo_b.unsqueeze(0)
            states.append(state)
        return torch.stack(states, dim=0)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Reconstruct the signal from Legendre coefficients.

        Parameters
        ----------
        coeffs: torch.Tensor
            Legendre coefficients of shape ``(..., order)``.

        Returns
        -------
        torch.Tensor
            Reconstructed signal of shape ``(..., eval_len)``.
        """
        eval_matrix = self.eval_matrix.to(dtype=coeffs.dtype)
        return (eval_matrix @ coeffs.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT, linear transform, inverse FFT.

    The layer transforms the input into the frequency domain, applies a
    learnable complex linear transform to the lowest ``modes`` frequency
    components and transforms the result back to the time domain.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    modes: int
        Number of lowest Fourier modes the transform is applied to.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes)
        )

    def _compl_mul1d(
        self,
        order: str,
        inputs: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> torch.Tensor:
        return torch.complex(
            torch.einsum(order, inputs.real, weights_real)
            - torch.einsum(order, inputs.imag, weights_imag),
            torch.einsum(order, inputs.real, weights_imag)
            + torch.einsum(order, inputs.imag, weights_real),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain filtering.

        Parameters
        ----------
        x: torch.Tensor
            Input of shape ``(batch_size, channels, features, seq_len)``.

        Returns
        -------
        torch.Tensor
            Filtered output of the same shape as the input when
            ``in_channels == out_channels``.
        """
        seq_len = x.size(-1)
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.out_channels,
            x_ft.size(-1),
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[..., :n_modes] = self._compl_mul1d(
            "bjix,iox->bjox",
            x_ft[..., :n_modes],
            self.weights_real[..., :n_modes],
            self.weights_imag[..., :n_modes],
        )
        return torch.fft.irfft(out_ft, n=seq_len, dim=-1)


class FiLM(TslibBaseModel):
    """FiLM: Frequency improved Legendre Memory model for forecasting.

    FiLM normalizes the input with a RevIN-style normalization, projects
    the recent history of several horizons onto Legendre polynomials with
    the HiPPO-LegT operator, filters the coefficients in the frequency
    domain with :class:`SpectralConv1d` layers, reconstructs the forecast
    horizons from the filtered coefficients and mixes the multiscale
    branches with a linear layer.

    Parameters
    ----------
    loss: nn.Module
        Loss function for training step.
    window_size: int, default=256
        Order of the HiPPO-LegT projection (number of Legendre
        polynomials).
    modes: int, default=32
        Maximum number of lowest Fourier modes kept in the spectral
        convolution layers.
    multiscale: tuple[int, ...] or list[int], default=(1, 2, 4)
        Multiples of ``prediction_length`` defining the input horizon of
        each spectral branch.
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler.
    metadata: Optional[dict], default=None
        Metadata for the model from TslibDataModule.

    References
    ----------
    [1] https://arxiv.org/abs/2205.08897
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py

    Notes
    -----
    [1] This implementation supports only continuous features. Categorical
        variables will be accommodated in future versions.
    [2] Only the forecasting task of the original implementation is ported.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.film._film_forecaster_v2 import (
            FiLMForecaster,
        )

        return FiLMForecaster

    def __init__(
        self,
        loss: nn.Module,
        window_size: int = 256,
        modes: int = 32,
        multiscale: tuple[int, ...] | list[int] = (1, 2, 4),
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

        warnings.warn(
            "FiLM is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution."
        )
        self.window_size = window_size
        self.modes = modes
        self.multiscale = multiscale

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialise the FiLM model network layer components."""
        self.enc_in = self.cont_dim + self.target_dim

        self.n_quantiles = None

        if isinstance(self._loss, QuantileLoss):
            self.n_quantiles = len(self._loss.quantiles)

        # RevIN-style per-channel affine parameters
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.enc_in))

        self.legts = nn.ModuleList(
            [
                HiPPOLegT(
                    order=self.window_size,
                    dt=1.0 / (self.prediction_length * scale),
                )
                for scale in self.multiscale
            ]
        )
        spectral_len = min(self.prediction_length, self.context_length)
        n_modes = max(1, min(self.modes, spectral_len // 2))
        self.spec_convs = nn.ModuleList(
            [
                SpectralConv1d(
                    in_channels=self.window_size,
                    out_channels=self.window_size,
                    modes=n_modes,
                )
                for _ in self.multiscale
            ]
        )
        self.mlp = nn.Linear(len(self.multiscale), 1)

        if self.n_quantiles is not None:
            self.quantile_head = nn.Linear(
                self.prediction_length,
                self.prediction_length * self.n_quantiles,
            )

    def _encoder(
        self, x: torch.Tensor, target_indices: torch.Tensor | None
    ) -> torch.Tensor:
        """Encode the input time series through the FiLM branches.

        Parameters
        ----------
        x: torch.Tensor
            Input data of shape (batch_size, context_length, n_features).
        target_indices: torch.Tensor or None
            Indices of target features to be extracted from the output.
            If None, all features are returned.

        Returns
        -------
        torch.Tensor
            Forecast of shape (batch_size, prediction_length, n_targets),
            or (batch_size, prediction_length, n_quantiles) for quantile
            losses with a single target.
        """
        # RevIN-style normalization
        means = x.mean(dim=1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(
            torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_norm = x_norm / stdev
        x_norm = x_norm * self.affine_weight + self.affine_bias

        branch_outputs = []
        for pos, scale in enumerate(self.multiscale):
            x_in = x_norm[:, -scale * self.prediction_length :]
            legt = self.legts[pos]
            # (batch, features, order, horizon)
            coeffs = legt(x_in.transpose(1, 2)).permute(1, 2, 3, 0)
            filtered = self.spec_convs[pos](coeffs)
            if self.context_length >= self.prediction_length:
                horizon_idx = self.prediction_length - 1
            else:
                horizon_idx = -1
            # (batch, features, order)
            selected = filtered.transpose(2, 3)[:, :, horizon_idx, :]
            # (batch, features, prediction_length)
            branch_outputs.append(
                selected @ legt.eval_matrix[-self.prediction_length :].T
            )

        stacked = torch.stack(branch_outputs, dim=-1)
        combined = self.mlp(stacked).squeeze(-1).transpose(1, 2)

        # RevIN-style de-normalization
        forecast = combined - self.affine_bias
        forecast = forecast / (self.affine_weight + 1e-10)
        forecast = forecast * stdev + means

        if target_indices is not None:
            forecast = forecast[:, :, target_indices]

        if self.n_quantiles is None:
            return forecast

        if forecast.shape[-1] != 1:
            raise ValueError(
                "Quantile forecasting currently only supports a single target."
            )
        batch_size = forecast.shape[0]
        return (
            self.quantile_head(forecast.transpose(1, 2))
            .transpose(1, 2)
            .reshape(batch_size, self.prediction_length, self.n_quantiles)
        )

    def _prepare_input_data(self, x: dict[str, torch.Tensor]):
        """Prepare input data and target indices for model input."""
        available_features = []
        current_idx = 0

        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])
            current_idx += x["history_cont"].size(-1)

        if "history_target" not in x or x["history_target"].size(-1) == 0:
            raise ValueError("No target history found in the input dictionary.")

        n_targets = x["history_target"].size(-1)
        target_indices = list(range(current_idx, current_idx + n_targets))
        available_features.append(x["history_target"])

        input_data = torch.cat(available_features, dim=-1)

        target_indices = torch.tensor(
            target_indices, dtype=torch.long, device=input_data.device
        )

        return input_data, target_indices

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the FiLM model.

        Parameters
        ----------
        x: dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors with
            ``prediction`` of shape
            (batch_size, prediction_length, target_dim).
        """
        input_data, target_indices = self._prepare_input_data(x)

        prediction = self._encoder(input_data, target_indices)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
