from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        layers = [
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
        self,
        backcast_theta: torch.Tensor,
        forecast_theta: torch.Tensor,
        encoder_x_t: torch.Tensor,
        decoder_x_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backcast = backcast_theta
        knots = forecast_theta

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )  # , align_corners=True)
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]

        return backcast, forecast


def init_weights(module, initialization):
    if type(module) is torch.nn.Linear:
        if initialization == "orthogonal":
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass  # torch.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel())) # noqa: E501
        else:
            assert 1 < 0, f"Initialization {initialization} not found"


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: list[int],
        activation: str,
        dropout: float,
    ):
        super().__init__()

        activ = getattr(nn, activation)()

        self.layers: nn.Sequential

        layers = [
            nn.Linear(in_features, hidden_size[0]),
        ]
        layers.append(activ)

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activ)

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_size[-1], out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


class NHiTSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        output_size: int,
        encoder_covariate_size: int,
        decoder_covariate_size: int,
        static_size: int,
        static_hidden_size: int,
        n_theta: int,
        hidden_size: list[int],
        pooling_sizes: int,
        pooling_mode: str,
        basis: nn.Module,
        n_layers: int,
        batch_normalization: bool,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        assert pooling_mode in ["max", "average"]

        self.context_length_pooled = int(np.ceil(context_length / pooling_sizes))

        if static_size == 0:
            static_hidden_size = 0

        self.context_length = context_length
        self.output_size = output_size
        self.n_theta = n_theta
        self.prediction_length = prediction_length
        self.static_size = static_size
        self.static_hidden_size = static_hidden_size
        self.encoder_covariate_size = encoder_covariate_size
        self.decoder_covariate_size = decoder_covariate_size
        self.pooling_sizes = pooling_sizes
        self.batch_normalization = batch_normalization
        self.dropout = dropout

        mlp_in_features = (
            self.context_length_pooled * len(self.output_size)
            + self.context_length * self.encoder_covariate_size
            + self.prediction_length * self.decoder_covariate_size
            + self.static_hidden_size
        )

        mlp_out_features = context_length * len(output_size) + n_theta * sum(
            output_size
        )

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"

        if pooling_mode == "max":
            self.pooling_layer = nn.MaxPool1d(
                kernel_size=self.pooling_sizes,
                stride=self.pooling_sizes,
                ceil_mode=True,
            )
        elif pooling_mode == "average":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.pooling_sizes,
                stride=self.pooling_sizes,
                ceil_mode=True,
            )

        # static_size is computed with data, static_hidden_size is provided by user,
        # if 0 no statics are used
        if (self.static_size > 0) and (self.static_hidden_size > 0):
            self.static_encoder = StaticFeaturesEncoder(
                in_features=static_size, out_features=static_hidden_size
            )

        self.layers = MLP(
            in_features=mlp_in_features,
            out_features=mlp_out_features,
            hidden_size=hidden_size,
            activation=activation,
            dropout=self.dropout,
        )

        self.basis = basis

    def forward(
        self,
        encoder_y: torch.Tensor,
        encoder_x_t: torch.Tensor,
        decoder_x_t: torch.Tensor,
        x_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encoder_y)

        encoder_y = encoder_y.transpose(1, 2)
        # Pooling layer to downsample input
        encoder_y = self.pooling_layer(encoder_y)
        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)

        if self.encoder_covariate_size > 0:
            encoder_y = torch.cat(
                (
                    encoder_y,
                    encoder_x_t.reshape(batch_size, -1),
                ),
                1,
            )

        if self.decoder_covariate_size > 0:
            encoder_y = torch.cat(
                (
                    encoder_y,
                    decoder_x_t.reshape(batch_size, -1),
                ),
                1,
            )

        # Static exogenous
        if (self.static_size > 0) and (self.static_hidden_size > 0):
            x_s = self.static_encoder(x_s)
            encoder_y = torch.cat((encoder_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(encoder_y)
        backcast_theta = theta[
            :, : self.context_length * len(self.output_size)
        ].reshape(-1, self.context_length)
        forecast_theta = theta[
            :, self.context_length * len(self.output_size) :
        ].reshape(-1, self.n_theta)
        backcast, forecast = self.basis(
            backcast_theta, forecast_theta, encoder_x_t, decoder_x_t
        )
        backcast = backcast.reshape(
            -1, len(self.output_size), self.context_length
        ).transpose(1, 2)
        forecast = forecast.reshape(
            -1, sum(self.output_size), self.prediction_length
        ).transpose(1, 2)

        return backcast, forecast


class NHiTS(nn.Module):
    """
    N-HiTS Model.
    """

    def __init__(
        self,
        context_length,
        prediction_length,
        output_size: int,
        static_size,
        encoder_covariate_size,
        decoder_covariate_size,
        static_hidden_size,
        n_blocks: list,
        n_layers: list,
        hidden_size: list,
        pooling_sizes: list,
        downsample_frequencies: list,
        pooling_mode,
        interpolation_mode,
        dropout,
        activation,
        initialization,
        batch_normalization,
        shared_weights,
        naive_level: bool,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.output_size = output_size
        self.naive_level = naive_level

        blocks = self.create_stack(
            n_blocks=n_blocks,
            context_length=context_length,
            prediction_length=prediction_length,
            output_size=output_size,
            encoder_covariate_size=encoder_covariate_size,
            decoder_covariate_size=decoder_covariate_size,
            static_size=static_size,
            static_hidden_size=static_hidden_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout=dropout,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(
        self,
        n_blocks,
        context_length,
        prediction_length,
        output_size,
        encoder_covariate_size,
        decoder_covariate_size,
        static_size,
        static_hidden_size,
        n_layers,
        hidden_size,
        pooling_sizes,
        downsample_frequencies,
        pooling_mode,
        interpolation_mode,
        batch_normalization,
        dropout,
        activation,
        shared_weights,
        initialization,
    ):
        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):
                # Batch norm only on first block
                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    n_theta = max(prediction_length // downsample_frequencies[i], 1)
                    basis = IdentityBasis(
                        backcast_size=context_length,
                        forecast_size=prediction_length,
                        interpolation_mode=interpolation_mode,
                    )

                    nbeats_block = NHiTSBlock(
                        context_length=context_length,
                        prediction_length=prediction_length,
                        output_size=output_size,
                        encoder_covariate_size=encoder_covariate_size,
                        decoder_covariate_size=decoder_covariate_size,
                        static_size=static_size,
                        static_hidden_size=static_hidden_size,
                        n_theta=n_theta,
                        hidden_size=hidden_size[i],
                        pooling_sizes=pooling_sizes[i],
                        pooling_mode=pooling_mode,
                        basis=basis,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout=dropout,
                        activation=activation,
                    )

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(
        self,
        encoder_y,
        encoder_mask,
        encoder_x_t,
        decoder_x_t,
        x_s,
    ):
        residuals = encoder_y  # .flip(dims=(1,))  # todo: check if flip is required or should be rather replaced by scatter # noqa: E501
        # encoder_x_t = encoder_x_t.flip(dims=(-1,))
        # encoder_mask = encoder_mask.flip(dims=(-1,))
        encoder_mask = encoder_mask.unsqueeze(-1)

        level = encoder_y[:, -1:].repeat(
            1, self.prediction_length, 1
        )  # Level with Naive1
        forecast_level = level.repeat_interleave(
            torch.tensor(self.output_size, device=level.device), dim=2
        )

        # level with last available observation
        if self.naive_level:
            block_forecasts = [forecast_level]
            block_backcasts = [encoder_y[:, -1:].repeat(1, self.context_length, 1)]

            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            block_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)

        # forecast by block
        for block in self.blocks:
            block_backcast, block_forecast = block(
                encoder_y=residuals,
                encoder_x_t=encoder_x_t,
                decoder_x_t=decoder_x_t,
                x_s=x_s,
            )
            residuals = (residuals - block_backcast) * encoder_mask

            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)

        # (n_batch, n_t, n_outputs, n_blocks)
        block_forecasts = torch.stack(block_forecasts, dim=-1)
        block_backcasts = torch.stack(block_backcasts, dim=-1)
        backcast = residuals

        return forecast, backcast, block_forecasts, block_backcasts
