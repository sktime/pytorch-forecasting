"""N-BEATS blocks conditioned on exogenous variables."""

import torch
from torch import nn

from pytorch_forecasting.layers._nbeats._blocks import SeasonalMixin, TrendMixin


def make_nbeatsx_blocks(
    hparams, input_size, context_length, prediction_length, exog_size
):
    """Resolve stack settings and build the residual blocks for either model API."""
    stack_types = hparams.stack_types
    if not stack_types or any(
        name not in {"generic", "trend", "seasonality", "exogenous"}
        for name in stack_types
    ):
        raise ValueError("stack_types must contain valid NBEATSx stack types")
    if not 0 <= hparams.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= hparams.backcast_loss_ratio < float("inf"):
        raise ValueError("backcast_loss_ratio must be finite and nonnegative")

    for name, default in (
        ("num_blocks", 3),
        ("num_block_layers", 4),
        ("widths", 256),
        ("sharing", False),
        ("expansion_coefficient_lengths", 3),
    ):
        values = hparams[name]
        if values is None:
            values = [default]
        if len(values) == 1:
            values = values * len(stack_types)
        if len(values) != len(stack_types):
            raise ValueError(f"{name} must have length 1 or len(stack_types)")
        if name != "sharing" and any(
            not isinstance(value, int) or value < 1 for value in values
        ):
            raise ValueError(f"{name} must contain positive integers")
        hparams[name] = values

    blocks = nn.ModuleList()
    for i, stack_type in enumerate(stack_types):
        for block_id in range(hparams.num_blocks[i]):
            if block_id > 0 and hparams.sharing[i]:
                block = blocks[-1]
            else:
                block = NBEATSxBlock(
                    stack_type=stack_type,
                    input_size=input_size,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    units=hparams.widths[i],
                    num_layers=hparams.num_block_layers[i],
                    expansion_coefficient_length=hparams.expansion_coefficient_lengths[
                        i
                    ],
                    exogenous_size=exog_size,
                    dropout=hparams.dropout,
                )
            blocks.append(block)
    return blocks


def run_nbeatsx_blocks(blocks, target, encoder, decoder, static, encoder_basis):
    """Return the forecast and backcast contributions of each block."""
    covariates = torch.cat([encoder.flatten(1), decoder.flatten(1), static], dim=-1)
    residual = target
    backcasts, forecasts = [], []
    for block in blocks:
        backcast, forecast = block(residual, covariates, encoder_basis, decoder)
        residual = residual - backcast
        backcasts.append(backcast)
        forecasts.append(forecast)
    return (
        target - residual,
        torch.stack(backcasts, dim=-1),
        torch.stack(forecasts, dim=-1),
    )


class NBEATSxBlock(nn.Module, TrendMixin, SeasonalMixin):
    """Project a target residual and covariates onto an N-BEATS basis.

    The exogenous basis uses the known temporal covariates directly, as in
    equation (10) of https://arxiv.org/abs/2104.05522. Historical-only and static
    covariates condition the projection coefficients without entering the basis.
    """

    def __init__(
        self,
        stack_type: str,
        input_size: int,
        context_length: int,
        prediction_length: int,
        units: int,
        num_layers: int,
        expansion_coefficient_length: int,
        exogenous_size: int,
        dropout: float,
    ):
        super().__init__()
        self.stack_type = stack_type
        layers = []
        for i in range(num_layers):
            layers.extend(
                [nn.Linear(input_size if i == 0 else units, units), nn.ReLU()]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)

        if stack_type == "trend":
            self._init_trend(
                context_length, prediction_length, expansion_coefficient_length
            )
            backcast_size = forecast_size = expansion_coefficient_length
        elif stack_type == "seasonality":
            self._init_seasonal(
                context_length,
                prediction_length,
                prediction_length,
                expansion_coefficient_length,
            )
            backcast_size = forecast_size = prediction_length
        elif stack_type == "exogenous":
            backcast_size = forecast_size = exogenous_size
        elif stack_type == "generic":
            backcast_size, forecast_size = context_length, prediction_length
        else:
            raise ValueError(f"Unknown stack type {stack_type!r}")
        self.theta_b_fc = nn.Linear(units, backcast_size)
        self.theta_f_fc = nn.Linear(units, forecast_size)

    def forward(self, residual, covariates, encoder_basis, decoder_basis):
        """Return backcast and forecast, each with shape (batch, time)."""
        hidden = self.fc(torch.cat([residual, covariates], dim=-1))
        if self.stack_type == "trend":
            return self.trend_forward(hidden, self.theta_b_fc, self.theta_f_fc)
        if self.stack_type == "seasonality":
            return self.seasonal_forward(hidden, self.theta_b_fc, self.theta_f_fc)
        backcast, forecast = self.theta_b_fc(hidden), self.theta_f_fc(hidden)
        if self.stack_type == "exogenous":
            backcast = torch.einsum("bc,btc->bt", backcast, encoder_basis)
            forecast = torch.einsum("bc,btc->bt", forecast, decoder_basis)
        return backcast, forecast
