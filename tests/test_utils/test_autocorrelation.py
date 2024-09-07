import math

import torch

from pytorch_forecasting.utils import autocorrelation


def test_autocorrelation():
    x = torch.sin(torch.linspace(0, 2 * 2 * math.pi, 201))
    corr = autocorrelation(x, dim=-1)
    assert corr[0] == 1, "Autocorrelation of first element should be 1."
    assert corr[101] > 0.99, "Autocorrelation should be near 1 for sin(2*pi)"
    assert corr[50] < -0.99, "Autocorrelation should be near -1 for sin(pi)"
