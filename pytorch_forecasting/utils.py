import os
from contextlib import redirect_stdout
from typing import Union, List, Dict, Tuple
import torch


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


def profile(function, profile_fname: str, filter: str = "", period=0.0001, **kwargs):
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
