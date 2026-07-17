import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting.layers._autoformer_encdec._series_decomp import series_decomp


class DFT_series_decomp(nn.Module):
    """Series decomposition using DFT"""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0  # zero out DC component

        top_k_freq, _ = torch.topk(freq, k=self.top_k, dim=1)
        threshold = top_k_freq.min(dim=1, keepdim=True).values
        xf = torch.where(freq >= threshold, xf, torch.zeros_like(xf))

        x_season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing for seasonal patterns"""

    def __init__(self, seq_len: int, down_sampling_window: int, num_down_sampling_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.num_down_sampling_layers = num_down_sampling_layers

        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(self.num_down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]

        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low

            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]

            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing for trend patterns"""

    def __init__(self, seq_len: int, down_sampling_window: int, num_down_sampling_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.num_down_sampling_layers = num_down_sampling_layers

        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(self.num_down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        trend_list_reverse = trend_list[::-1]

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]

        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high

            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]

            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """Main decomposable multi-scale mixing block"""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        down_sampling_window: int,
        num_down_sampling_layers: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        channel_independence: bool,
        decomp_method: str = "moving_avg",
        moving_avg: int | None = None,
        top_k: int | None = None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.num_down_sampling_layers = num_down_sampling_layers
        self.channel_independence = channel_independence

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if decomp_method == "moving_avg":
            if moving_avg is None:
                raise ValueError("moving_avg must be provided")
            self.decomposition = series_decomp(moving_avg)

        elif decomp_method == "dft_decomp":
            if top_k is None:
                raise ValueError("top_k must be provided")
            self.decomposition = DFT_series_decomp(top_k)

        else:
            raise ValueError("Invalid decomposition method")

        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            num_down_sampling_layers=self.num_down_sampling_layers,
        )

        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            num_down_sampling_layers=self.num_down_sampling_layers,
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]

        season_list = []
        trend_list = []

        for x in x_list:
            season, trend = self.decomposition(x)

            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)

            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend

            if self.channel_independence:
                out = ori + self.out_cross_layer(out)

            out_list.append(out[:, :length, :])

        return out_list