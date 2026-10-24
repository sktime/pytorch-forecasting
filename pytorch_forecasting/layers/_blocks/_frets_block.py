"""Core FreTS frequency-domain MLP layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreTSCore(nn.Module):
    """Core FreTS algorithm.

    Frequency-domain MLP for time series forecasting.
    Applies FFT along time and optionally channel dimensions,
    learns patterns via diagonal complex MLPs, then decodes
    with a fully-connected layer.

    Parameters
    ----------
    context_length : int
        Length of the encoder input sequence.
    prediction_length : int
        Number of future steps to predict.
    n_channels : int
        Number of input channels (continuous features).
    embed_size : int, default=128
        Dimension of the token embedding.
    hidden_size : int, default=256
        Hidden size of the FC output head.
    channel_independence : bool, default=True
        If False, apply cross-channel frequency mixing before
        temporal frequency mixing.
    sparsity_threshold : float, default=0.01
        Soft-shrinkage threshold applied to frequency coefficients
        to promote sparsity.
    scale : float, default=0.02
        Initialisation scale for complex MLP weight parameters.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_channels: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        channel_independence: bool = True,
        sparsity_threshold: float = 0.01,
        scale: float = 0.02,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_channels = n_channels
        self.embed_size = embed_size
        self.channel_independence = channel_independence
        self.sparsity_threshold = sparsity_threshold

        self.embeddings = nn.Parameter(torch.randn(1, embed_size))

        self.r1 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.i1 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.rb1 = nn.Parameter(scale * torch.randn(embed_size))
        self.ib1 = nn.Parameter(scale * torch.randn(embed_size))

        self.r2 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.i2 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.rb2 = nn.Parameter(scale * torch.randn(embed_size))
        self.ib2 = nn.Parameter(scale * torch.randn(embed_size))

        self.fc = nn.Sequential(
            nn.Linear(context_length * embed_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, prediction_length),
        )

    def _token_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Expand each scalar time step into embed_size dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, N)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, T, embed_size)``.
        """
        x = x.permute(0, 2, 1).unsqueeze(3)
        return x * self.embeddings

    def _fre_mlp(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        i: torch.Tensor,
        rb: torch.Tensor,
        ib: torch.Tensor,
    ) -> torch.Tensor:
        """Diagonal complex-valued MLP in the frequency domain.

        The ``"bijd,dd->bijd"`` contraction repeats ``d`` on both operands and on
        the output, so it is an element-wise multiplication by ``diag(r)`` rather
        than a dense matrix product: only the diagonal of ``r`` and ``i`` takes
        part in the computation. This mirrors the reference FreTS implementation
        (https://github.com/aikunyi/FreTS) and is kept as-is so that results stay
        comparable with the paper.

        Parameters
        ----------
        x : torch.Tensor
            Complex tensor of shape ``(B, nd, dim//2+1, embed_size)``.
        r : torch.Tensor
            Real weight matrix of shape ``(embed_size, embed_size)``. Only its
            diagonal is used, see the note above.
        i : torch.Tensor
            Imaginary weight matrix of shape ``(embed_size, embed_size)``. Only
            its diagonal is used, see the note above.
        rb : torch.Tensor
            Real bias of shape ``(embed_size,)``.
        ib : torch.Tensor
            Imaginary bias of shape ``(embed_size,)``.

        Returns
        -------
        torch.Tensor
            Complex tensor of the same shape as ``x``.
        """
        o_real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, r)
            - torch.einsum("bijd,dd->bijd", x.imag, i)
            + rb
        )
        o_imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, r)
            + torch.einsum("bijd,dd->bijd", x.real, i)
            + ib
        )
        y = torch.stack([o_real, o_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)

    def _mlp_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Frequency temporal learner: FFT along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, N, T, embed_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, T, embed_size)``.
        """
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        y = self._fre_mlp(x, self.r2, self.i2, self.rb2, self.ib2)
        return torch.fft.irfft(y, n=self.context_length, dim=2, norm="ortho")

    def _mlp_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Frequency channel learner: FFT along the channel dimension.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, N, T, embed_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, T, embed_size)``.
        """
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        y = self._fre_mlp(x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.n_channels, dim=2, norm="ortho")
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FreTS core.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, N)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, prediction_length, N)``.
        """
        B, _, N = x.shape

        x = self._token_emb(x)
        bias = x

        if not self.channel_independence:
            x = self._mlp_channel(x)

        x = self._mlp_temporal(x)
        x = x + bias

        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
