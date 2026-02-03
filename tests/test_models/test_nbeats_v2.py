import torch

from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.nbeats import NBEATS
from pytorch_forecasting.models.nbeats._nbeats_v2_pkg import NBEATS_pkg_v2


def _make_metadata(context_length=10, prediction_length=5):
    return {
        "max_encoder_length": context_length,
        "max_prediction_length": prediction_length,
    }


def test_nbeats_v2_forward_shapes():
    batch_size = 4
    context_length = 10
    prediction_length = 5

    model = NBEATS(metadata=_make_metadata(context_length, prediction_length))
    model.eval()

    x = {
        "target_past": torch.randn(batch_size, context_length, 1),
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
    model = NBEATS(metadata=_make_metadata())
    model.eval()

    batch_size = 2

    x = {
        "target_past": torch.randn(batch_size, model.context_length, 1),
    }

    out = model(x)

    for key in ["trend", "seasonality", "generic"]:
        assert key in out
        assert out[key].dim() == 3


def test_nbeats_v2_training_step_and_backward():
    batch_size = 3
    context_length = 8
    prediction_length = 4

    model = NBEATS(
        loss=MAE(),
        metadata=_make_metadata(context_length, prediction_length),
    )
    model.train()

    x = {
        "target_past": torch.randn(batch_size, context_length, 1),
    }

    y = torch.randn(batch_size, prediction_length)

    out = model.training_step((x, y), batch_idx=0)

    assert isinstance(out, dict)
    assert "loss" in out
    assert torch.is_tensor(out["loss"])
    assert out["loss"].requires_grad

    out["loss"].backward()


def test_nbeats_v2_pkg_get_cls():
    cls = NBEATS_pkg_v2.get_cls()
    assert cls is NBEATS


def test_nbeats_v2_pkg_test_params():
    params = NBEATS_pkg_v2.get_test_params()
    assert isinstance(params, dict)
