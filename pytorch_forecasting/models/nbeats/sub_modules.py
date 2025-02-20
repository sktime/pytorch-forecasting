"""
Implementation of ``nn.Modules`` for N-Beats model.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.models.nbeats.kan_layer import KANLayer


def linear(input_size, output_size, bias=True, dropout: int = None):
    """
    Initialize linear layers for MLP block layers.
    """
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


def linspace(
    backcast_length: int, forecast_length: int, centered: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate linear spaced values for backcast and forecast.
    """
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(
        start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32
    )
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
        dropout=0.1,
        kan_params=None,
        use_kan=False,
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers.
            thetas_dim: The dimension of the parameterized output for the block.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units
                from the past are used to predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps
                ahead to predict. Default: 5.
            dropout: The dropout rate applied to the fully connected mlp layers to
                prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer
                (used for modeling using KAN). Default: None.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by
                        KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1,
                        the grid is uniform; if 0, grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as
                        a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
            use_kan: flag parameter to decide usage of KAN blocks in NBEATS. if true,
                kan layers are used in nbeats block else mlp layers are used. Default:
                false.
        """
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.kan_params = kan_params
        self.use_kan = use_kan

        if self.use_kan:
            layers = [
                KANLayer(
                    in_dim=backcast_length,
                    out_dim=units,
                    **self.kan_params,
                )
            ]

            # Add additional layers for deeper structure
            for _ in range(num_block_layers - 1):
                layers.append(
                    KANLayer(
                        in_dim=units,
                        out_dim=units,
                        **self.kan_params,
                    )
                )

            # Define the fully connected layers
            self.fc = nn.Sequential(*layers)

            # Define the theta layers
            self.theta_f_fc = self.theta_b_fc = KANLayer(
                in_dim=units,
                out_dim=thetas_dim,
                **self.kan_params,
            )

        else:
            fc_stack = [
                nn.Linear(backcast_length, units),
                nn.ReLU(),
            ]
            for _ in range(num_block_layers - 1):
                fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
            self.fc = nn.Sequential(*fc_stack)
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        """
        Pass through the fully connected mlp/kan layers and returns the output.
        """
        if self.use_kan:
            # save outputs to be used in updating grid in kan layers during training
            # outputs logic taken from
            # https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L2682
            self.outputs = []
            self.outputs.append(x.clone().detach())
            for layer in self.fc:
                x = layer(x)  # Pass data through the current layer
                # storing outputs for updating grids of self.fc when using KAN
                self.outputs.append(x.clone().detach())
            # storing for updating grids of theta_b_fc and theta_f_fc when using KAN
            self.outputs.append(x.clone().detach())
            return x  # Return final output
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
        kan_params=None,
        use_kan=False,
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers.
            thetas_dim: The dimension of the parameterized output for the block.
                If None, it is inferred.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units
                from the past are used to predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps
                ahead to predict. Default: 5.
            nb_harmonics: The number of harmonics in the seasonal function (relevant for
                seasonal models). Default: None (no seasonality).
            min_period: The minimum period used for seasonal patterns. Default: 1.
            dropout: The dropout rate applied to the fully connected mlp layers to
                prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer
                (used for modeling using KAN). Default: None.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by
                        KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1,
                        the grid is uniform; if 0, grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as
                        a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
            use_kan: flag parameter to decide usage of KAN blocks in NBEATS. if true,
                kan layers are used in nbeats block else mlp layers are used. Default:
                false.
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
            dropout=dropout,
            kan_params=kan_params,
            use_kan=use_kan,
        )

        backcast_linspace, forecast_linspace = linspace(
            backcast_length, forecast_length, centered=False
        )

        p1, p2 = (
            (thetas_dim // 2, thetas_dim // 2)
            if thetas_dim % 2 == 0
            else (thetas_dim // 2, thetas_dim // 2 + 1)
        )
        s1_b = torch.tensor(
            np.cos(2 * np.pi * self.get_frequencies(p1)[:, None] * backcast_linspace),
            dtype=torch.float32,
        )  # H/2-1
        s2_b = torch.tensor(
            np.sin(2 * np.pi * self.get_frequencies(p2)[:, None] * backcast_linspace),
            dtype=torch.float32,
        )
        self.register_buffer("S_backcast", torch.cat([s1_b, s2_b]))

        s1_f = torch.tensor(
            np.cos(2 * np.pi * self.get_frequencies(p1)[:, None] * forecast_linspace),
            dtype=torch.float32,
        )  # H/2-1
        s2_f = torch.tensor(
            np.sin(2 * np.pi * self.get_frequencies(p2)[:, None] * forecast_linspace),
            dtype=torch.float32,
        )
        self.register_buffer("S_forecast", torch.cat([s1_f, s2_f]))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the backcast and forecast outputs for the given input tensor.
        """
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
        return np.linspace(
            0, (self.backcast_length + self.forecast_length) / self.min_period, n
        )


class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
        kan_params=None,
        use_kan=False,
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers.
            thetas_dim: The dimension of the parameterized output for the block.
                If None, it is inferred.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units
                from the past are used to predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps
                ahead to predict. Default: 5.
            dropout: The dropout rate applied to the fully connected mlp layers to
                prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer
                (used for modeling using KAN). Default: None.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by
                        KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1,
                        the grid is uniform; if 0, grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as
                        a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
            use_kan: flag parameter to decide usage of KAN blocks in NBEATS. if true,
                kan layers are used in nbeats block else mlp layers are used. Default:
                false.
        """
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            kan_params=kan_params,
            use_kan=use_kan,
        )

        backcast_linspace, forecast_linspace = linspace(
            backcast_length, forecast_length, centered=True
        )
        norm = np.sqrt(
            forecast_length / thetas_dim
        )  # ensure range of predictions is comparable to input
        thetas_dims_range = np.array(range(thetas_dim))
        coefficients = torch.tensor(
            backcast_linspace ** thetas_dims_range[:, None],
            dtype=torch.float32,
        )
        self.register_buffer("T_backcast", coefficients * norm)
        coefficients = torch.tensor(
            forecast_linspace ** thetas_dims_range[:, None],
            dtype=torch.float32,
        )
        self.register_buffer("T_forecast", coefficients * norm)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the backcast and forecast outputs for the given input tensor.
        """
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
        kan_params=None,
        use_kan=False,
    ):
        """
        Initialize NBeatsSeasonalBlock

        Args:
            units: The number of units in the mlp/kan layers.
            thetas_dim: The dimension of the parameterized output for the block.
                If None, it is inferred.
            num_block_layers: Number of fully connected mlp/kan layers. Default: 4.
            backcast_length: The length of the backcast. Defines how many time units
                from the past are used to predict the future. Default: 10.
            forecast_length: The length of the forecast, i.e., the number of time steps
                ahead to predict. Default: 5.
            dropout: The dropout rate applied to the fully connected mlp layers to
                prevent overfitting. Default: 0.1.
            kan_params (dict): Parameters specific to the KAN layer
                (used for modeling using KAN). Default: None.
                Contains:
                    num_grids (int): The number of grid intervals for KAN.
                    k (int): The order of the piecewise polynomial for KAN.
                    noise_scale (float): The scale of noise injected at initialization.
                    scale_base_mu (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_base_sigma (float): The scale of the residual function
                        initialized to N(scale_base_mu, scale_base_sigma^2).
                    scale_sp (float): The scale of the base function spline(x) in KAN.
                    base_fun (function): The residual function used by
                        KAN (e.g., torch.nn.SiLU()).
                    grid_eps (float): Determines the partitioning of the grid. If 1,
                        the grid is uniform; if 0, grid is partitioned by percentiles.
                    grid_range (list or np.array): The range of the grid, given as
                        a list of two values.
                    sp_trainable (bool): If True, the scale_sp is trainable.
                    sb_trainable (bool): If True, the scale_base is trainable.
                    sparse_init (bool): If True, applies sparse initialization.
            use_kan: flag parameter to decide usage of KAN blocks in NBEATS. if true,
                kan layers are used in nbeats block else mlp layers are used. Default:
                false.
        """
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            kan_params=kan_params,
            use_kan=use_kan,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        """
        Computes the backcast and forecast outputs for the given input tensor.
        """
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
