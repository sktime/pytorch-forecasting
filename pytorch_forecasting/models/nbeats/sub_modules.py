"""
Implementation of ``nn.Modules`` for N-Beats model.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear(input_size, output_size, bias=True, dropout: int = None):
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


def linspace(backcast_length: int, forecast_length: int, centered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        share_thetas=False,
        dropout=0.1,
    ):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas

        fc_stack = [
            nn.Linear(backcast_length, units),
            nn.ReLU(),
        ]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
        self.fc = nn.Sequential(*fc_stack)

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


class NBEATSSeasonalBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim=None,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
        min_period=1,
        dropout=0.1,
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        self.min_period = min_period

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=False)

        p1, p2 = (thetas_dim // 2, thetas_dim // 2) if thetas_dim % 2 == 0 else (thetas_dim // 2, thetas_dim // 2 + 1)
        s1_b = torch.tensor(
            [np.cos(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p1)], dtype=torch.float32
        )  # H/2-1
        s2_b = torch.tensor(
            [np.sin(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p2)], dtype=torch.float32
        )
        self.register_buffer("S_backcast", torch.cat([s1_b, s2_b]))

        s1_f = torch.tensor(
            [np.cos(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p1)], dtype=torch.float32
        )  # H/2-1
        s2_f = torch.tensor(
            [np.sin(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p2)], dtype=torch.float32
        )
        self.register_buffer("S_forecast", torch.cat([s1_f, s2_f]))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        amplitudes_backward = self.theta_b_fc(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        amplitudes_forward = self.theta_f_fc(x)
        forecast = amplitudes_forward.mm(self.S_forecast)

        return backcast, forecast

    def get_frequencies(self, n):
        return np.linspace(0, (self.backcast_length + self.forecast_length) / self.min_period, n)


class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=True)
        norm = np.sqrt(forecast_length / thetas_dim)  # ensure range of predictions is comparable to input

        coefficients = torch.tensor([backcast_linspace ** i for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer("T_backcast", coefficients * norm)

        coefficients = torch.tensor([forecast_linspace ** i for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer("T_forecast", coefficients * norm)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
