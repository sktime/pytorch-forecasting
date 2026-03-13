import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):
    """Multi-scale 2D convolutional block for temporal pattern extraction.

    Runs ``num_kernels`` parallel ``Conv2d`` layers whose kernel sizes are
    ``1x1``, ``3x3``, ``5x5``, … (i.e. ``2*i+1`` for i = 0 … num_kernels-1),
    each with matching padding so that the spatial dimensions are preserved.
    The outputs are stacked along a new last dimension and averaged, giving a
    single output tensor of shape ``(B, out_channels, H, W)``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels produced by every Conv2d kernel.
    num_kernels : int, default=6
        How many parallel convolution kernels to use.
    init_weight : bool, default=True
        If ``True``, initialise Conv2d weights with Kaiming-normal and zero
        biases.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, out_channels, H, W)`` — average of all kernel outputs.
        """
        res_list = [kernel(x) for kernel in self.kernels]
        return torch.stack(res_list, dim=-1).mean(dim=-1)


def FFT_for_Period(x: torch.Tensor, k: int = 2):
    """Detect the top-k dominant periods in a multivariate time series.

    Uses the real-valued FFT along the time axis to compute per-frequency
    amplitudes, then returns the ``k`` frequencies with the largest average
    amplitude (excluding the DC component) together with their amplitudes.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, T, C)``.
    k : int, default=2
        Number of top periods (frequencies) to return.

    Returns
    -------
    period : numpy.ndarray
        Array of shape ``(k,)`` containing the period lengths corresponding to
        the top-k frequencies.  Period length = T // frequency_index.
    period_weight : torch.Tensor
        Tensor of shape ``(B, k)`` containing the mean amplitude across all
        channels for each of the top-k frequencies.
    """
    xf = torch.fft.rfft(x, dim=1)

    frequency_list = abs(xf).mean(dim=0).mean(dim=-1)
    frequency_list[0] = 0

    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    period = x.shape[1] // top_list  # numpy array of shape (k,)

    period_weight = abs(xf).mean(dim=-1)[:, top_list]

    return period, period_weight


class TimesBlock(nn.Module):
    """Core building block of TimesNet.

    For each of the ``top_k`` dominant periods detected by FFT:

    1. Pad the sequence so its length is an exact multiple of the period.
    2. Reshape the padded 1-D sequence into a 2-D grid of shape
       ``(length // period, period)`` — one "row" per cycle.
    3. Apply two stacked ``Inception_Block_V1`` layers (with GELU in between)
       to capture multi-scale 2-D patterns.
    4. Reshape back to the original 1-D sequence length.

    The ``top_k`` results are adaptively aggregated with softmax weights
    derived from the FFT amplitudes.  A residual connection adds the original
    input back to the aggregated output.

    Parameters
    ----------
    seq_len : int
        Encoder context length ``T``.
    pred_len : int
        Forecast horizon ``H``.
    top_k : int
        Number of dominant periods to detect via FFT.
    d_model : int
        Model hidden dimension (number of features / channels along the ``N``
        axis of the sequence tensor).
    d_ff : int
        Inner (feedforward) dimension of the Inception blocks.
    num_kernels : int
        Number of parallel Conv2d kernels inside each ``Inception_Block_V1``.

    Notes
    -----
    ``TimesBlock`` always operates on a sequence whose length equals
    ``seq_len + pred_len`` — the temporal extension is applied *before* the
    first TimesBlock by ``TimesNetModel.predict_linear``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T+H, d_model)``.  The sequence has already been
            extended from ``T`` to ``T+H`` by ``TimesNetModel.predict_linear``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T+H, d_model)`` — same as input (residual output).
        """
        B, T, N = x.size()
        full_len = self.seq_len + self.pred_len

        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            if full_len % period != 0:
                padded_len = ((full_len // period) + 1) * period
                pad_size = padded_len - full_len
                padding = torch.zeros(B, pad_size, N, dtype=x.dtype, device=x.device)
                out = torch.cat([x, padding], dim=1)  # (B, padded_len, N)
            else:
                padded_len = full_len
                out = x  # (B, full_len, N)

            out = (
                out.reshape(B, padded_len // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            out = self.conv(out)  # (B, N, padded_len//period, period)

            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :full_len, :])  # (B, full_len, N)

        res = torch.stack(res, dim=-1)  # (B, full_len, N, k)

        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).expand(B, T, N, self.k)
        res = torch.sum(res * period_weight, dim=-1)  # (B, full_len, N)

        res = res + x
        return res
