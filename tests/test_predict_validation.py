"""Tests for input validation and argument handling in the v2 predict path."""

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_forecasting.callbacks.predict import PredictCallback
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.dlinear._dlinear_forecaster_v2 import (
    DLinearForecaster,
)
from pytorch_forecasting.utils._validation import _check_choice, _check_choices

_VALID_KEYS = ["x", "y", "index", "decoder_lengths"]

_TRAINER_KWARGS = {
    "accelerator": "cpu",
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


class _TinyModel(BaseModel):
    """Smallest possible v2 model, only used to exercise ``predict``."""

    def forward(self, x):
        return {"prediction": x["v"][:, :2].unsqueeze(-1)}


def _dataloader(n_samples=6, batch_size=2):
    generator = torch.Generator().manual_seed(0)
    samples = [
        (
            {
                "v": torch.randn(3, generator=generator),
                "decoder_lengths": torch.tensor(2),
            },
            torch.randn(2, generator=generator),
        )
        for _ in range(n_samples)
    ]
    return DataLoader(samples, batch_size=batch_size)


def _model():
    return _TinyModel(loss=MAE())


# ---------------------------------------------------------------- helpers


def test_check_choice_accepts_a_valid_value():
    _check_choice("a", ["a", "b"], "param")


def test_check_choice_names_the_value_and_the_options():
    message = r"`param` must be one of \['a', 'b'\], got 'c'"
    with pytest.raises(ValueError, match=message):
        _check_choice("c", ["a", "b"], "param")


def test_check_choice_does_not_accept_a_list_of_valid_values():
    with pytest.raises(ValueError, match="`param` must be one of"):
        _check_choice(["a"], ["a", "b"], "param")


def test_check_choices_accepts_valid_and_empty_values():
    _check_choices(["a", "b"], ["a", "b"], "param")
    _check_choices([], ["a", "b"], "param")


def test_check_choices_lists_only_the_unknown_values():
    with pytest.raises(ValueError, match=r"unknown value\(s\) \['c', 'd'\]"):
        _check_choices(["a", "c", "d"], ["a", "b"], "param")


# ---------------------------------------------------------------- callback


@pytest.mark.parametrize("mode", ["prediction", "quantiles", "raw"])
def test_callback_accepts_every_valid_mode(mode):
    assert PredictCallback(mode=mode).mode == mode


@pytest.mark.parametrize("mode", ["bogus", "Prediction", None, ["prediction"]])
def test_callback_rejects_an_invalid_mode(mode):
    with pytest.raises(ValueError, match="`mode` must be one of"):
        PredictCallback(mode=mode)


@pytest.mark.parametrize("return_info", [None, [], _VALID_KEYS, tuple(_VALID_KEYS)])
def test_callback_accepts_valid_return_info(return_info):
    callback = PredictCallback(return_info=return_info)
    assert callback.return_info == list(return_info or [])


def test_callback_rejects_an_unknown_return_info_key():
    with pytest.raises(ValueError, match=r"`return_info`.*\['not_a_real_key'\]"):
        PredictCallback(return_info=["x", "not_a_real_key"])


def test_callback_treats_a_single_string_as_one_key():
    assert PredictCallback(return_info="index").return_info == ["index"]


def test_callback_rejects_a_single_unknown_string():
    with pytest.raises(ValueError, match=r"\['indx'\]"):
        PredictCallback(return_info="indx")


@pytest.mark.parametrize("return_info", [5, {"index"}, {"index": True}])
def test_callback_rejects_return_info_of_the_wrong_type(return_info):
    with pytest.raises(TypeError, match="`return_info` must be list or tuple"):
        PredictCallback(return_info=return_info)


def test_callback_keeps_its_own_copy_of_return_info():
    keys = ["index"]
    callback = PredictCallback(return_info=keys)
    keys.append("x")
    assert callback.return_info == ["index"]


# ---------------------------------------------------------------- BaseModel


def test_predict_fails_before_any_batch_on_unknown_return_info():
    """The error is raised up front, not after the whole prediction pass."""
    seen_batches = []

    class _Counting(_TinyModel):
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            seen_batches.append(batch_idx)
            return super().predict_step(batch, batch_idx, dataloader_idx)

    model = _Counting(loss=MAE())
    with pytest.raises(ValueError, match="not_a_real_key"):
        model.predict(
            _dataloader(),
            return_info=["not_a_real_key"],
            trainer_kwargs=dict(_TRAINER_KWARGS),
        )
    assert seen_batches == []


def test_predict_does_not_modify_trainer_kwargs():
    trainer_kwargs = dict(_TRAINER_KWARGS)
    expected = dict(trainer_kwargs)

    model = _model()
    for _ in range(2):
        model.predict(_dataloader(), trainer_kwargs=trainer_kwargs)

    assert trainer_kwargs == expected
    assert "callbacks" not in trainer_kwargs


@pytest.mark.parametrize("wrap", [list, tuple])
def test_predict_does_not_modify_the_callbacks_of_the_caller(wrap):
    own_callback = PredictCallback()
    callbacks = wrap([own_callback])
    trainer_kwargs = {**_TRAINER_KWARGS, "callbacks": callbacks}

    model = _model()
    for _ in range(2):
        model.predict(_dataloader(), trainer_kwargs=trainer_kwargs)

    assert trainer_kwargs["callbacks"] is callbacks
    assert list(callbacks) == [own_callback]


def test_predict_still_runs_the_callbacks_of_the_caller():
    own_callback = PredictCallback()
    trainer_kwargs = {**_TRAINER_KWARGS, "callbacks": own_callback}

    _model().predict(_dataloader(), trainer_kwargs=trainer_kwargs)

    assert trainer_kwargs["callbacks"] is own_callback
    assert own_callback.result["prediction"].shape == (6, 2)


def test_repeated_predict_calls_give_the_same_result():
    model = _model()
    dataloader = _dataloader()
    trainer_kwargs = dict(_TRAINER_KWARGS)

    first = model.predict(dataloader, trainer_kwargs=trainer_kwargs)
    second = model.predict(dataloader, trainer_kwargs=trainer_kwargs)

    assert first["prediction"].shape == (6, 2)
    assert torch.equal(first["prediction"], second["prediction"])


def test_predict_returns_the_requested_return_info():
    out = _model().predict(
        _dataloader(),
        return_info=["x", "decoder_lengths"],
        trainer_kwargs=dict(_TRAINER_KWARGS),
    )
    assert set(out) == {"prediction", "x", "decoder_lengths"}
    assert out["x"]["v"].shape == (6, 3)
    assert out["decoder_lengths"].shape == (6,)


# ---------------------------------------------------------------- forecaster


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"mode": "bogus"}, "`mode` must be one of"),
        ({"return_info": ["not_a_real_key"]}, "not_a_real_key"),
    ],
)
def test_forecaster_predict_fails_before_loading_the_data(kwargs, error):
    """``BaseForecaster.predict`` checks its arguments before any data work."""
    forecaster = DLinearForecaster()
    forecaster.model = object()  # anything but None, the model is not reached

    def _fail(*args, **kwargs):
        raise AssertionError("data was loaded before the arguments were checked")

    forecaster._load_dataloader = _fail

    with pytest.raises(ValueError, match=error):
        forecaster.predict(data=None, **kwargs)
