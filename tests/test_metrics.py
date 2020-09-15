import torch
from pytorch_forecasting.metrics import CompositeMetric, SMAPE, MAE


def test_composite_metric():
    metric1 = SMAPE()
    metric2 = MAE()
    combined_metric = 0.3 * metric1 + 2.0 * metric2
    assert isinstance(combined_metric, CompositeMetric), "combined metric should be composite metric"

    # test repr()
    repr(combined_metric)

    # test results
    y = torch.normal(0, 1, (10, 20)).abs()
    y_pred = torch.normal(0, 1, (10, 20)).abs()

    res1 = metric1(y_pred, y)
    res2 = metric2(y_pred, y)
    combined_res = combined_metric(y_pred, y)

    assert torch.isclose(combined_res, res1 * 0.3 + res2 * 2.0)

    # test quantiles and prediction
    combined_metric.to_prediction(y_pred)
    combined_metric.to_quantiles(y_pred)