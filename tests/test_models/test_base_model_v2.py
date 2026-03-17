import pytest
import torch
from torch import nn

from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear

_METADATA = {
    "feature_names": {
        "categorical": [],
        "continuous": [],
        "static": [],
        "known": [],
        "unknown": [],
        "target": ["value"],
        "all": [],
        "static_categorical": [],
        "static_continuous": [],
    },
    "feature_indices": {
        "categorical": [],
        "continuous": [],
        "static": [],
        "known": [],
        "unknown": [],
        "target": [0],
    },
    "n_features": {
        "categorical": 0,
        "continuous": 0,
        "static": 0,
        "known": 0,
        "unknown": 0,
        "target": 1,
        "all": 0,
        "static_categorical": 0,
        "static_continuous": 0,
    },
    "context_length": 16,
    "prediction_length": 4,
    "freq": "h",
    "features": "S",
}


@pytest.fixture
def model_with_logging_metrics():
    with pytest.warns(UserWarning):
        model = DLinear(
            loss=MAE(),
            logging_metrics=[SMAPE(), MAE()],
            metadata=_METADATA,
        )
    return model


def test_logging_metrics_is_module_list(model_with_logging_metrics):
    assert isinstance(model_with_logging_metrics.logging_metrics, nn.ModuleList)


def test_logging_metrics_device_propagation(model_with_logging_metrics):
    model_with_logging_metrics.to("meta")
    for metric in model_with_logging_metrics.logging_metrics:
        for state_name in metric._defaults:
            val = getattr(metric, state_name)
            if isinstance(val, torch.Tensor):
                assert val.device.type == "meta"
