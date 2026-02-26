"""
Helper functions for PyTorch forecasting
"""

from collections import namedtuple
from collections.abc import Callable
from contextlib import redirect_stdout
import inspect
import os
from typing import Any, Union

import lightning.pytorch as pl
import torch
from torch import nn
from torch.fft import irfft, rfft
import torch.nn.functional as F
from torch.nn.utils import rnn


def integer_histogram(
    data: torch.LongTensor, min: None | int = None, max: None | int = None
) -> torch.Tensor:
    """
    Create histogram of integers in predefined range.

    Parameters
    ----------
    data : torch.LongTensor
        Data for which to create histogram.
    min : int, optional
        Minimum of histogram, is inferred from data by default.
    max : int, optional
        Maximum of histogram, is inferred from data by default.

    Returns
    -------
    torch.Tensor
        Histogram.
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
    keys: torch.Tensor,
    values: torch.Tensor,
    bins: int = 95,
    reduction: str = "mean",
    return_histogram: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Groupby apply for torch tensors.

    Parameters
    ----------
    keys : torch.Tensor
        Tensor of groups (``0`` to ``bins``).
    values : torch.Tensor
        Values to aggregate - same size as keys.
    bins : int, optional
        Total number of groups. Defaults to 95.
    reduction : str, optional
        Either "mean" or "sum". Defaults to "mean".
    return_histogram : bool, optional
        If to return histogram on top. Defaults to False.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Tensor of size ``bins`` with aggregated values
        and optionally with counts of values.
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    uniques, counts = keys.unique(return_counts=True)
    groups = torch.stack(
        [reduce(item) for item in torch.split_with_sizes(values, tuple(counts))]
    )
    reduced = torch.zeros(bins, dtype=values.dtype, device=values.device).scatter(
        dim=0, index=uniques, src=groups
    )
    if return_histogram:
        hist = torch.zeros(bins, dtype=torch.long, device=values.device).scatter(
            dim=0, index=uniques, src=counts
        )
        return reduced, hist
    else:
        return reduced


def profile(
    function: Callable, profile_fname: str, filter: str = "", period=0.0001, **kwargs
):
    """
    Profile a given function with ``vmprof``.

    Parameters
    ----------
    function : Callable
        Function to profile.
    profile_fname : str
        Path where to save profile (`.txt` file will be saved with line profile).
    filter : str, optional
        Filter name (e.g. module name) to filter profile. Defaults to "".
    period : float, optional
        Frequency of calling profiler in seconds. Defaults to 0.0001.
    """  # noqa : E501
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

    Parameters
    ----------
    n : int
        Number of classes.
    max_size : int, optional
        Maximum embedding size. Defaults to 100.

    Returns
    -------
    int
        Embedding size.
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


def create_mask(
    size: int, lengths: torch.LongTensor, inverse: bool = False
) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lengths) x size.

    An entry at (i, j) is True if lengths[i] > j.

    Parameters
    ----------
    size : int
        Size of second dimension.
    lengths : torch.LongTensor
        Tensor of lengths.
    inverse : bool, optional
        If true, boolean mask is inverted. Defaults to False.

    Returns
    -------
    torch.BoolTensor
        Mask tensor.
    """

    if inverse:  # return where values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
    else:  # return where no values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(-1)


_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro.

    Parameters
    ----------
    size : int
        A positive number.

    Returns
    -------
    int
        A possibly larger number.
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

    Implementation copied from `pyro <https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py>`_.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    dim : int, optional
        The dimension to calculate autocorrelation. Defaults to 0.

    Returns
    -------
    torch.Tensor
        Autocorrelation of ``input``.
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
    autocorr = autocorr / torch.tensor(
        range(N, 0, -1), dtype=input.dtype, device=input.device
    )
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


def unpack_sequence(
    sequence: torch.Tensor | rnn.PackedSequence,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack RNN sequence.

    Parameters
    ----------
    sequence : torch.Tensor or rnn.PackedSequence
        RNN packed sequence or tensor of which first index are samples and
        second are timesteps.

    Returns
    -------
    tuple of torch.Tensor
        Tuple of unpacked sequence and length of samples.
    """  # noqa : E501
    if isinstance(sequence, rnn.PackedSequence):
        sequence, lengths = rnn.pad_packed_sequence(sequence, batch_first=True)
        # batch sizes reside on the CPU by default -> we need to bring them to GPU
        lengths = lengths.to(sequence.device)
    else:
        lengths = torch.ones(
            sequence.size(0), device=sequence.device, dtype=torch.long
        ) * sequence.size(1)
    return sequence, lengths


def concat_sequences(
    sequences: list[torch.Tensor] | list[rnn.PackedSequence],
) -> torch.Tensor | rnn.PackedSequence:
    """
    Concatenate RNN sequences.

    Parameters
    ----------
    sequences : list of torch.Tensor or list of rnn.PackedSequence
        List of RNN packed sequences or tensors of which first index are samples
        and second are timesteps.

    Returns
    -------
    torch.Tensor or rnn.PackedSequence
        Concatenated sequence.
    """  # noqa : E501
    if isinstance(sequences[0], rnn.PackedSequence):
        return rnn.pack_sequence(sequences, enforce_sorted=False)
    elif isinstance(sequences[0], torch.Tensor):
        return torch.cat(sequences, dim=0)
    elif isinstance(sequences[0], tuple | list):
        return tuple(
            concat_sequences([sequences[ii][i] for ii in range(len(sequences))])
            for i in range(len(sequences[0]))
        )
    else:
        raise ValueError("Unsupported sequence type")


def padded_stack(
    tensors: list[torch.Tensor],
    side: str = "right",
    mode: str = "constant",
    value: int | float = 0,
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Parameters
    ----------
    tensors : list of torch.Tensor
        List of tensors to stack.
    side : str, optional
        Side on which to pad - "left" or "right". Defaults to "right".
    mode : str, optional
        'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
    value : int or float, optional
        Value to use for constant padding. Defaults to 0.

    Returns
    -------
    torch.Tensor
        Stacked tensor.
    """  # noqa : E501
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
            (
                F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value)
                if full_size - x.size(-1) > 0
                else x
            )
            for x in tensors
        ],
        dim=0,
    )
    return out


def to_list(value: Any) -> list[Any]:
    """
    Convert value or list to list of values.
    If already list, return object directly.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    list of Any
        List of values.
    """
    if isinstance(value, tuple | list) and not isinstance(value, rnn.PackedSequence):
        return value
    else:
        return [value]


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to unsqueeze.
    like : torch.Tensor
        Tensor whose dimensions to match.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]


def apply_to_list(obj: list[Any] | Any, func: Callable) -> list[Any] | Any:
    """
    Apply function to a list of objects or directly if passed value is not a list.

    This is useful if the passed object could be either a list to whose elements
    a function needs to be applied or just an object to which to apply the function.

    Parameters
    ----------
    obj : list of Any or Any
        List/tuple on whose elements to apply function, otherwise
        object to whom to apply function.
    func : Callable
        Function to apply.

    Returns
    -------
    list of Any or Any
        List of objects or object depending on function output
        and if input ``obj`` is of type list/tuple.
    """
    if isinstance(obj, tuple | list) and not isinstance(obj, rnn.PackedSequence):
        return [func(o) for o in obj]
    else:
        return func(obj)


class OutputMixIn:
    """
    MixIn to give namedtuple some access capabilities of a dictionary.
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        else:
            return super().__getitem__(k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def items(self):
        return zip(self._fields, self)

    def keys(self):
        return self._fields

    def iget(self, idx: int | slice):
        """
        Select item(s) row-wise.

        Parameters
        ----------
        idx : int or slice
            Item to select.

        Returns
        -------
        Any
            Output of single item.
        """
        return self.__class__(*(x[idx] for x in self))


class TupleOutputMixIn:
    """MixIn to give output a namedtuple-like access capabilities with ``to_network_output() function``."""  # noqa : E501

    def to_network_output(self, **results):
        """
        Convert output into a named (and immutable) tuple.

        This allows tracing the modules as graphs and prevents modifying the output.

        Returns
        -------
        namedtuple
            Network output as a named tuple.
        """
        if hasattr(self, "_output_class"):
            Output = self._output_class
        else:
            OutputTuple = namedtuple("output", results)

            class Output(OutputMixIn, OutputTuple):
                pass

            self._output_class = Output

        return self._output_class(**results)


def move_to_device(
    x: dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor],
    device: str | torch.DeviceObjType,
) -> (
    dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
):
    """
    Move object to device.

    Parameters
    ----------
    x : dict, list, tuple, or torch.Tensor
        Object (e.g. dictionary) of tensors to move to device.
    device : str or torch.DeviceObjType
        Device, e.g. "cpu".

    Returns
    -------
    dict, list, tuple, or torch.Tensor
        Input `x` on targeted device.
    """  # noqa: E501
    if isinstance(device, str):
        if device == "mps":
            if hasattr(torch.backends, device):
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
        else:
            device = torch.device(device)
    if isinstance(x, dict):
        for name in x.keys():
            x[name] = move_to_device(x[name], device=device)
    elif isinstance(x, OutputMixIn):
        for xi in x:
            move_to_device(xi, device=device)
        return x
    elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device)
    elif isinstance(x, tuple | list) and x[0].device != device:
        x = [move_to_device(xi, device=device) for xi in x]
    return x


def detach(
    x: dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor],
) -> (
    dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
):
    """
    Detach object.

    Parameters
    ----------
    x : dict, list, tuple, or torch.Tensor
        Object to detach.

    Returns
    -------
    dict, list, tuple, or torch.Tensor
        Detached object.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, dict):
        return {name: detach(xi) for name, xi in x.items()}
    elif isinstance(x, OutputMixIn):
        return x.__class__(**{name: detach(xi) for name, xi in x.items()})
    elif isinstance(x, tuple | list):
        return [detach(xi) for xi in x]
    else:
        return x


def masked_op(
    tensor: torch.Tensor, op: str = "mean", dim: int = 0, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculate operation on masked tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to conduct operation over.
    op : str, optional
        Operation to apply. One of ["mean", "sum"]. Defaults to "mean".
    dim : int, optional
        Dimension to average over. Defaults to 0.
    mask : torch.Tensor, optional
        Boolean mask to apply (True=will take mean, False=ignore).
        Masks nan values by default.

    Returns
    -------
    torch.Tensor
        Tensor with averaged out dimension.
    """  # noqa : E501
    if mask is None:
        mask = ~torch.isnan(tensor)
    masked = tensor.masked_fill(~mask, 0.0)
    summed = masked.sum(dim=dim)
    if op == "mean":
        return summed / mask.sum(dim=dim)  # Find the average
    elif op == "sum":
        return summed
    else:
        raise ValueError(f"unknown operation {op}")


def repr_class(
    obj,
    attributes: list[str] | dict[str, Any],
    max_characters_before_break: int = 100,
    extra_attributes: dict[str, Any] = None,
) -> str:
    """
    Print class name and parameters.

    Parameters
    ----------
    obj : Any
        Class to format.
    attributes : list of str or dict of str to Any
        List of attributes to show or dictionary of attributes and values to show.
    max_characters_before_break : int, optional
        Number of characters before breaking into multiple lines. Defaults to 100.
    extra_attributes : dict of str to Any, optional
        Extra attributes to show in angled brackets.

    Returns
    -------
    str
        Formatted string representation of the class.
    """  # noqa E501
    if extra_attributes is None:
        extra_attributes = {}
    # get attributes
    if isinstance(attributes, tuple | list):
        attributes = {
            name: getattr(obj, name) for name in attributes if hasattr(obj, name)
        }
    attributes_strings = [f"{name}={repr(value)}" for name, value in attributes.items()]
    # get header
    header_name = obj.__class__.__name__
    # add extra attributes
    if len(extra_attributes) > 0:
        extra_attributes_strings = [
            f"{name}={repr(value)}" for name, value in extra_attributes.items()
        ]
        if (
            len(header_name) + 2 + len(", ".join(extra_attributes_strings))
            > max_characters_before_break
        ):
            header = f"{header_name}[\n\t" + ",\n\t".join(attributes_strings) + "\n]("
        else:
            header = f"{header_name}[{', '.join(extra_attributes_strings)}]("
    else:
        header = f"{header_name}("

    # create final representation
    attributes_string = ", ".join(attributes_strings)
    if (
        len(attributes_string) + len(header.split("\n")[-1]) + 1
        > max_characters_before_break
    ):
        attributes_string = "\n\t" + ",\n\t".join(attributes_strings) + "\n"
    return f"{header}{attributes_string})"


class InitialParameterRepresenterMixIn:
    def __repr__(self) -> str:
        if isinstance(self, nn.Module):
            return super().__repr__()
        else:
            attributes = list(inspect.signature(self.__class__).parameters.keys())
            return repr_class(self, attributes=attributes)

    def extra_repr(self) -> str:
        """
        Return extra information about parameters for representation/logging.
        """
        if isinstance(self, pl.LightningModule):
            return "\t" + repr(self.hparams).replace("\n", "\n\t")
        else:
            attributes = list(inspect.signature(self.__class__).parameters.keys())
            return ", ".join(
                [
                    f"{name}={repr(getattr(self, name))}"
                    for name in attributes
                    if hasattr(self, name)
                ]
            )
