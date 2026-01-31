import torch

from pytorch_forecasting.metrics import MASE
from pytorch_forecasting.models.nbeats import NBEATS_v2
from pytorch_forecasting.models.nbeats._nbeats_pkg_v2 import NBEATS_pkg_v2


def _make_metadata(context_length=10, prediction_length=5):
    return {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "feature_indices": {
            "target": [0],
        },
        "n_features": {
            "target": 1,
        },
    }


def test_nbeats_v2_forward_shapes():
    batch_size = 4
    context_length = 10
    prediction_length = 5

    model = NBEATS_v2(metadata=_make_metadata(context_length, prediction_length))
    model.eval()

    x = {
        "target": torch.randn(batch_size, context_length),
    }

    out = model(x)

    assert "prediction" in out
    assert "backcast" in out

    assert out["prediction"].shape == (
        batch_size,
        prediction_length,
        1,
    )

    assert out["backcast"].shape == (
        batch_size,
        context_length,
        1,
    )


def test_nbeats_v2_decomposition_outputs_exist():
    model = NBEATS_v2(metadata=_make_metadata())
    model.eval()

    x = {
        "target": torch.randn(2, model.context_length),
    }

    out = model(x)

    assert "trend" in out
    assert "seasonality" in out

    assert out["trend"].dim() == 3
    assert out["seasonality"].dim() == 3


def test_nbeats_v2_training_step_and_backward():
    batch_size = 3
    context_length = 8
    prediction_length = 4

    model = NBEATS_v2(
        loss=MASE(),
        metadata=_make_metadata(context_length, prediction_length),
    )
    model.train()

    x = {
        "target": torch.randn(batch_size, context_length),
    }
    y = torch.randn(batch_size, prediction_length, 1)

    loss = model.training_step((x, y), batch_idx=0)

    assert torch.is_tensor(loss)
    assert loss.requires_grad

    loss.backward()


def test_nbeats_v2_pkg_get_cls():
    cls = NBEATS_pkg_v2.get_cls()
    assert cls is NBEATS_v2


def test_nbeats_v2_pkg_test_params():
    params = NBEATS_pkg_v2.get_base_test_params()

    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, dict) for p in params)
