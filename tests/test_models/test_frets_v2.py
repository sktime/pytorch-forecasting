import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.frets._frets_pkg_v2 import FreTS_pkg_v2
from pytorch_forecasting.models.frets._frets_v2 import FreTS

CONTEXT_LENGTH = 6
PREDICTION_LENGTH = 3
BATCH_SIZE = 4
N_SERIES = 3
N_SAMPLES = 80

# fast, deterministic trainer config for the integration tests
TRAINER_CFG = {
    "max_epochs": 1,
    "limit_train_batches": 2,
    "limit_val_batches": 1,
    "enable_checkpointing": False,
    "logger": False,
    "accelerator": "cpu",
}


@pytest.fixture
def test_data():
    """Create synthetic ``TimeSeries`` data for fit/predict.

    Returns
    -------
    dict
        Dictionary with ``"train"`` and ``"predict"`` ``TimeSeries`` datasets,
        mirroring the structure consumed by the v2 package ``fit``/``predict``
        API.
    """
    rng = np.random.default_rng(0)
    time_idx = np.arange(N_SAMPLES)
    series_data = []
    for i in range(N_SERIES):
        values = np.sin(2 * np.pi * time_idx / 20) + rng.normal(0, 0.1, N_SAMPLES)
        series_data.append(
            pd.DataFrame({"time_idx": time_idx, "series_id": i, "value": values})
        )
    data = pd.concat(series_data).reset_index(drop=True)

    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=[],
        unknown=["value"],
    )
    return {"train": ts, "predict": ts}


def _build_pkg(model_cfg):
    """Build a ``FreTS_pkg_v2`` from a ``get_test_train_params`` entry.

    Parameters
    ----------
    model_cfg : dict
        One entry returned by
        :meth:`FreTS_pkg_v2.get_test_train_params`. The ``"datamodule_cfg"``
        key is split out and forwarded to the datamodule; everything else is
        passed to the model.

    Returns
    -------
    FreTS_pkg_v2
        Configured package instance, ready for ``fit``.
    """
    model_cfg = dict(model_cfg)
    dm_cfg = dict(model_cfg.pop("datamodule_cfg"))
    dm_cfg.setdefault("batch_size", BATCH_SIZE)
    # ensure a non-empty model_cfg so the package can build from scratch
    model_cfg.setdefault("loss", SMAPE())
    return FreTS_pkg_v2(
        model_cfg=model_cfg,
        trainer_cfg=TRAINER_CFG,
        datamodule_cfg=dm_cfg,
    )


@pytest.mark.parametrize("model_cfg", FreTS_pkg_v2.get_test_train_params())
def test_frets_v2_integration(test_data, model_cfg):
    """End-to-end fit + predict through the package for each test config.

    This drives the model the same way the generic v2 estimator suite does
    (``pkg.fit`` then ``pkg.predict``), and covers every configuration in
    ``get_test_train_params`` -- including both ``channel_independence``
    modes, the ``embed_size``/``hidden_size`` variants and the default loss.

    Parameters
    ----------
    test_data : dict
        Fixture providing ``"train"``/``"predict"`` ``TimeSeries``.
    model_cfg : dict
        A single configuration from ``get_test_train_params``.
    """
    pkg = _build_pkg(model_cfg)
    expected_pred_len = pkg.datamodule_cfg["max_prediction_length"]

    pkg.fit(test_data["train"], save_ckpt=False)
    predictions = pkg.predict(test_data["predict"], mode="raw")

    assert predictions is not None
    assert isinstance(predictions, dict)
    assert "prediction" in predictions

    pred = predictions["prediction"]
    assert isinstance(pred, torch.Tensor)
    assert pred.ndim == 3, f"prediction must be 3D, got {pred.ndim}D"
    assert pred.shape[1] == expected_pred_len


def test_frets_v2_pkg_get_cls():
    """Test that ``FreTS_pkg_v2.get_cls()`` returns ``FreTS``."""
    assert FreTS_pkg_v2.get_cls() is FreTS


def test_frets_v2_pkg_naming_convention():
    """Test that pkg class name follows the convention ``<model>_pkg_v2``."""
    model_cls = FreTS_pkg_v2.get_cls()
    expected_pkg_name = model_cls.__name__ + "_pkg_v2"
    assert FreTS_pkg_v2.__name__ == expected_pkg_name


def test_frets_v2_pkg_test_train_params():
    """Test that ``get_test_train_params`` returns a non-empty list of dicts."""
    params = FreTS_pkg_v2.get_test_train_params()
    assert isinstance(params, list)
    assert len(params) > 0
    for p in params:
        assert isinstance(p, dict)
        assert "datamodule_cfg" in p
