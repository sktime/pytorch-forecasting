"""
Helper functions for PyTorch forecasting
"""
from contextlib import redirect_stdout
import os
from typing import Callable, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import rnn


def integer_histogram(
    data: torch.LongTensor, min: Union[None, int] = None, max: Union[None, int] = None
) -> torch.Tensor:
    """
    Create histogram of integers in predefined range

    Args:
        data: data for which to create histogram
        min: minimum of histogram, is inferred from data by default
        max: maximum of histogram, is inferred from data by default

    Returns:
        histogram
    """
    uniques, counts = torch.unique(data, return_counts=True)
    if min is None:
        min = uniques.min()
    if max is None:
        max = uniques.max()
    hist = torch.zeros(max - min + 1, dtype=torch.long, device=data.device).scatter(
        dim=0, index=uniques - min, src=counts
    )
    return hist


def groupby_apply(
    keys: torch.Tensor, values: torch.Tensor, bins: int = 95, reduction: str = "mean", return_histogram: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Groupby apply for torch tensors

    Args:
        keys: tensor of groups (``0`` to ``bins``)
        values: values to aggregate - same size as keys
        bins: total number of groups
        reduction: either "mean" or "sum"
        return_histogram: if to return histogram on top

    Returns:
        tensor of size ``bins`` with aggregated values and optionally with counts of values
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    uniques, counts = keys.unique(return_counts=True)
    groups = torch.stack([reduce(item) for item in torch.split_with_sizes(values, tuple(counts))])
    reduced = torch.zeros(bins, dtype=values.dtype, device=values.device).scatter(dim=0, index=uniques, src=groups)
    if return_histogram:
        hist = torch.zeros(bins, dtype=torch.long, device=values.device).scatter(dim=0, index=uniques, src=counts)
        return reduced, hist
    else:
        return reduced


def profile(function: Callable, profile_fname: str, filter: str = "", period=0.0001, **kwargs):
    """
    Profile a given function with ``vmprof``.

    Args:
        function (Callable): function to profile
        profile_fname (str): path where to save profile (`.txt` file will be saved with line profile)
        filter (str, optional): filter name (e.g. module name) to filter profile. Defaults to "".
        period (float, optional): frequency of calling profiler in seconds. Defaults to 0.0001.
    """
    import vmprof
    from vmprof.show import LinesPrinter

    # profiler config
    with open(profile_fname, "wb+") as fd:
        # start profiler
        vmprof.enable(fd.fileno(), lines=True, period=period)
        # run function
        function(**kwargs)
        # stop profiler
        vmprof.disable()

    # write report to disk
    if kwargs.get("lines", True):
        with open(f"{os.path.splitext(profile_fname)[0]}.txt", "w") as f:
            with redirect_stdout(f):
                LinesPrinter(filter=filter).show(profile_fname)


def get_embedding_size(n: int) -> int:
    if n > 2:
        return round(1.6 * n ** 0.56)
    else:
        return 1


_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    Implementation copied form ``pyro``.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    if (not input.is_cuda) and (not torch.backends.mkl.is_available()):
        raise NotImplementedError("For CPU tensor, this method is only supported " "with MKL installed.")

    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    # centering and padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)
    pad = torch.zeros(input.shape[:-1] + (M2 - N,), dtype=input.dtype, device=input.device)
    centered_signal = torch.cat([centered_signal, pad], dim=-1)

    # Fourier transform
    freqvec = torch.rfft(centered_signal, signal_ndim=1, onesided=False)
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1, keepdim=True)
    freqvec_gram = torch.cat(
        [freqvec_gram, torch.zeros(freqvec_gram.shape, dtype=input.dtype, device=input.device)], dim=-1
    )
    # inverse Fourier transform
    autocorr = torch.irfft(freqvec_gram, signal_ndim=1, onesided=False)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(range(N, 0, -1), dtype=input.dtype, device=input.device)
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


def unpack_sequence(sequence: Union[torch.Tensor, rnn.PackedSequence]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack RNN sequence.

    Args:
        sequence (Union[torch.Tensor, rnn.PackedSequence]): RNN packed sequence or tensor of which
            first index are samples and second are timesteps

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of unpacked sequence and length of samples
    """
    if isinstance(sequence, rnn.PackedSequence):
        sequence, lengths = rnn.pad_packed_sequence(sequence, batch_first=True)
        # batch sizes reside on the CPU by default -> we need to bring them to GPU
        lengths = lengths.to(sequence.device)
    else:
        lengths = torch.ones(sequence.size(0), device=sequence.device, dtype=torch.long) * sequence.size(1)
    return sequence, lengths


def padded_stack(
    tensors: List[torch.Tensor], side: str = "right", mode: str = "constant", value: Union[int, float] = 0
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out
