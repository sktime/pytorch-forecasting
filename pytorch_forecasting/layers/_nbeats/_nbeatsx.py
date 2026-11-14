"""N-BEATS blocks conditioned on exogenous variables."""

import torch
from torch import nn

from pytorch_forecasting.layers._nbeats._blocks import SeasonalMixin, TrendMixin


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
