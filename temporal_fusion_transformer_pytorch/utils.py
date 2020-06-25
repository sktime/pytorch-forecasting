import os
from contextlib import redirect_stdout
from typing import Union, List, Dict
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
    hist = torch.zeros(max - min + 1, dtype=torch.long).scatter(dim=0, index=uniques - min, src=counts)
    return hist


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
