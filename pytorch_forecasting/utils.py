"""
Helper functions for PyTorch forecasting
"""
from contextlib import redirect_stdout
import os
from typing import Any, Callable, List, Tuple, Union

import torch
from torch.fft import irfft, rfft
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


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n ** 0.56), max_size)
    else:
        return 1


def create_mask(size: int, lengths: torch.LongTensor, inverse: bool = False) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lenghts) x size.

    An entry at (i, j) is True if lengths[i] > j.

    Args:
        size (int): size of second dimension
        lengths (torch.LongTensor): tensor of lengths
        inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

    Returns:
        torch.BoolTensor: mask
    """

    if inverse:  # return where values are
        return torch.arange(size, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(-1)
    else:  # return where no values are
        return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)


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

    Implementation copied form `pyro <https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py>`_.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    # centering and padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

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


def to_list(value: Any) -> List[Any]:
    """
    Convert value or list to list of values.
    If already list, return object directly

    Args:
        value (Any): value to convert

    Returns:
        List[Any]: list of values
    """
    if isinstance(value, (tuple, list)) and not isinstance(value, rnn.PackedSequence):
        return value
    else:
        return [value]


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]


def apply_to_list(obj: Union[List[Any], Any], func: Callable) -> Union[List[Any], Any]:
    """
    Apply function to a list of objects or directly if passed value is not a list.

    This is useful if the passed object could be either a list to whose elements
    a function needs to be applied or just an object to whicht to apply the function.

    Args:
        obj (Union[List[Any], Any]): list/tuple on whose elements to apply function,
            otherwise object to whom to apply function
        func (Callable): function to apply

    Returns:
        Union[List[Any], Any]: list of objects or object depending on function output and
            if input ``obj`` is of type list/tuple
    """
    if isinstance(obj, (list, tuple)) and not isinstance(obj, rnn.PackedSequence):
        return [func(o) for o in obj]
    else:
        return func(obj)
