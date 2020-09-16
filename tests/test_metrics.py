import itertools

import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting.metrics import MAE, SMAPE, AggregationMetric, CompositeMetric


def test_composite_metric():
    metric1 = SMAPE()
    metric2 = MAE()
    combined_metric = 1.0 * (0.3 * metric1 + 2.0 * metric2 + metric1)
    assert isinstance(combined_metric, CompositeMetric), "combined metric should be composite metric"

    # test repr()
    repr(combined_metric)

    # test results
    y = torch.normal(0, 1, (10, 20)).abs()
    y_pred = torch.normal(0, 1, (10, 20)).abs()

    res1 = metric1(y_pred, y)
    res2 = metric2(y_pred, y)
    combined_res = combined_metric(y_pred, y)

    assert torch.isclose(combined_res, res1 * 0.3 + res2 * 2.0 + res1)

    # test quantiles and prediction
    combined_metric.to_prediction(y_pred)
    combined_metric.to_quantiles(y_pred)


@pytest.mark.parametrize(
    "decoder_lengths,y",
    [
        (torch.tensor([1, 2], dtype=torch.long), torch.tensor([[0.0, 1.0], [5.0, 1.0]])),
        (2 * torch.ones(2, dtype=torch.long), torch.tensor([[0.0, 1.0], [5.0, 1.0]])),
        (2 * torch.ones(2, dtype=torch.long), torch.tensor([[[0.0, 1.0], [1.0, 1.0]], [[5.0, 1.0], [1.0, 2.0]]])),
    ],
)
def test_aggregation_metric(decoder_lengths, y):
    y_pred = torch.tensor([[0.0, 2.0], [4.0, 3.0]])
    if (decoder_lengths != y_pred.size(-1)).any():
        y_packed = rnn.pack_padded_sequence(y, lengths=decoder_lengths, batch_first=True, enforce_sorted=False)
    else:
        y_packed = y

    # metric
    metric = AggregationMetric(MAE())
    res = metric(y_pred, y_packed)
    if (decoder_lengths == y_pred.size(-1)).all() and y.ndim == 2:
        assert torch.isclose(res, (y.mean(0) - y_pred.mean(0)).abs().mean())
