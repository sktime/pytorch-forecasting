"""
Implementation of ``nn.Modules`` for N-Beats model.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_layer import KANLayer


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


# class NBEATSBlock(nn.Module):
#     def __init__(
#         self,
#         units,
#         thetas_dim,
#         num_block_layers=4,
#         backcast_length=10,
#         forecast_length=5,
#         share_thetas=False,
#         num_grid_intervals=5,
#         k_order=3,
#         dropout=0.1,
#     ):
#         super().__init__()
#         self.units = units
#         self.thetas_dim = thetas_dim
#         self.backcast_length = backcast_length
#         self.forecast_length = forecast_length
#         self.share_thetas = share_thetas
#         # First KANLayer
#         layers = [
#             KANLayer(
#                 in_dim=backcast_length,
#                 out_dim=units,
#                 num=num_grid_intervals,
#                 k=k_order,
#                 device="cpu",
#             )
#         ]
#         # Additional KANLayers for deeper structure
#         for _ in range(num_block_layers - 1):
#             layers.extend(
#                 [
#                     KANLayer(
#                         in_dim=units,
#                         out_dim=units,
#                         num=num_grid_intervals,
#                         k=k_order,
#                         device="cpu",
#                     )
#                 ]
#             )
#         self.fc = nn.Sequential(*layers)
#         # print(self.fc)
#         # Theta layers
#         if share_thetas:
#             self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
#         else:
#             self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
#             self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

#     def forward(self, x):
#         # x = x.unsqueeze(0)
#         # print(x.shape,"here")
#         y = self.fc(x)
#         # print("bhen")
#         return y


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
        kan_params={},
    ):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.kan_params = kan_params

        if self.kan_params["use_kan"]:
            layers = [
                KANLayer(
                    in_dim=backcast_length,
                    out_dim=units,
                    num=self.kan_params["num_grids"],
                    k=self.kan_params["k"],
                    noise_scale=self.kan_params["noise_scale"],
                    scale_base_mu=self.kan_params["scale_base_mu"],
                    scale_base_sigma=self.kan_params["scale_base_sigma"],
                    scale_sp=self.kan_params["scale_sp"],
                    base_fun=self.kan_params["base_fun"],
                    grid_eps=self.kan_params["grid_eps"],
                    grid_range=self.kan_params["grid_range"],
                    sp_trainable=self.kan_params["sp_trainable"],
                    sb_trainable=self.kan_params["sb_trainable"],
                    sparse_init=self.kan_params["sparse_init"],
                )
            ]
            # Additional KANLayers for deeper structure
            for _ in range(num_block_layers - 1):
                layers.extend(
                    [
                        KANLayer(
                            in_dim=units,
                            out_dim=units,
                            num=self.kan_params["num_grids"],
                            k=self.kan_params["k"],
                            noise_scale=self.kan_params["noise_scale"],
                            scale_base_mu=self.kan_params["scale_base_mu"],
                            scale_base_sigma=self.kan_params["scale_base_sigma"],
                            scale_sp=self.kan_params["scale_sp"],
                            base_fun=self.kan_params["base_fun"],
                            grid_eps=self.kan_params["grid_eps"],
                            grid_range=self.kan_params["grid_range"],
                            sp_trainable=self.kan_params["sp_trainable"],
                            sb_trainable=self.kan_params["sb_trainable"],
                            sparse_init=self.kan_params["sparse_init"],
                            device="cpu",  # Assuming you are using the "cpu" device
                        )
                    ]
                )

        self.fc = nn.Sequential(*layers)

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
        kan_params={},
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers. Default: 256.
            thetas_dim: The dimension of the parameterized output for the block. If None, it is inferred. Default: None.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units from the past are used to
                predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps ahead to predict. Default: 5.
            nb_harmonics: The number of harmonics in the seasonal function (relevant for seasonal models).
                Default: None (no seasonality).
            min_period: The minimum period used for seasonal patterns. Default: 1.
            dropout: The dropout rate applied to the fully connected mlp layers to prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer (used for modeling using KAN).
                Default: empty dictionary.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function initialized to
                        N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function initialized to
                        N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1, the grid is uniform; if 0,
                        grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
        """
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
            kan_params=kan_params,
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
        """Computes the backcast and forecast outputs for the given input tensor."""
        x = super().forward(x)
        amplitudes_backward = self.theta_b_fc(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        amplitudes_forward = self.theta_f_fc(x)
        forecast = amplitudes_forward.mm(self.S_forecast)

        return backcast, forecast

    def get_frequencies(self, n):
        """
        Generates frequency values based on the backcast and forecast lengths.
        """
        return np.linspace(0, (self.backcast_length + self.forecast_length) / self.min_period, n)


class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, dropout=0.1, kan_params={}
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers. Default: 256.
            thetas_dim: The dimension of the parameterized output for the block. If None, it is inferred. Default: None.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units from the past are used to
                predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps ahead to predict. Default: 5.
            dropout: The dropout rate applied to the fully connected mlp layers to prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer (used for modeling using KAN).
                Default: empty dictionary.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function initialized to
                        N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function initialized to
                        N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1, the grid is uniform; if 0,
                        grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
        """
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
            kan_params=kan_params,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=True)
        norm = np.sqrt(forecast_length / thetas_dim)  # ensure range of predictions is comparable to input

        coefficients = torch.tensor([backcast_linspace**i for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer("T_backcast", coefficients * norm)

        coefficients = torch.tensor([forecast_linspace**i for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer("T_forecast", coefficients * norm)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):
    def __init__(
        self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, dropout=0.1, kan_params={}
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers. Default: 256.
            thetas_dim: The dimension of the parameterized output for the block. If None, it is inferred. Default: None.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units from the past are used to
                predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps ahead to predict. Default: 5.
            dropout: The dropout rate applied to the fully connected mlp layers to prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer (used for modeling using KAN).
                Default: empty dictionary.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function initialized to
                        N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function initialized to
                        N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1, the grid is uniform; if 0,
                        grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
        """
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            kan_params=kan_params,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
