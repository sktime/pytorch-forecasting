"""TimesBlock and FFT period discovery module for TimesNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers._convolution._inception_block import InceptionBlockV1


def fft_for_period(x: torch.Tensor, k: int = 2) -> tuple[list[int], torch.Tensor]:
    """Find the top-k dominant periods in a batch of time series via FFT.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(batch_size, seq_len, channels)``.
    k : int, default=2
        Maximum number of dominant periods to extract.

    Returns
    -------
    periods : list[int]
        List of discovered period lengths (at most ``k``).
    amplitudes : torch.Tensor
        Mean FFT amplitudes for each selected frequency, of shape
        ``(batch_size, len(periods))``.
    """
    seq_len = x.shape[1]
    xf = torch.fft.rfft(x, dim=1)

    frequency_list = abs(xf).mean(0).mean(-1)

    if frequency_list.shape[0] <= 1:
        return [1], abs(xf).mean(-1)[:, :1]

    actual_k = min(k, frequency_list.shape[0] - 1)
    _, top_indices = torch.topk(frequency_list[1:], actual_k)
    top_indices = top_indices + 1

    periods = [max(1, seq_len // int(idx)) for idx in top_indices.detach().cpu()]
    amplitudes = abs(xf).mean(-1)[:, top_indices]
    return periods, amplitudes


class TimesBlock(nn.Module):
    """Core TimesNet block: 1D -> 2D folding, 2D Inception CNN, and unfolding.

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int
        Intermediate channel dimension inside the 2D Inception block pair.
    top_k : int, default=5
        Number of dominant FFT periods to fold and process per block.
    num_kernels : int, default=6
        Number of parallel 2D convolution kernels in each Inception block.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int = 5,
        num_kernels: int = 6,
    ):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a 1D sequence tensor of shape ``(B, T, d_model)``."""
        batch_size, seq_len, channels = x.size()
        period_list, period_weight = fft_for_period(x, self.top_k)

        res = []
        for period in period_list:
            if seq_len % period != 0:
                length = ((seq_len // period) + 1) * period
                padding = torch.zeros(
                    (batch_size, length - seq_len, channels),
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = seq_len
                out = x

            out = (
                out.reshape(batch_size, length // period, period, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)

            # Unfold 2D -> 1D and remove padding: [B, T, C]
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            res.append(out[:, :seq_len, :])

        res_tensor = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = (
            period_weight.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, channels, 1)
        )
        out = torch.sum(res_tensor * period_weight, dim=-1)

        return out + x
