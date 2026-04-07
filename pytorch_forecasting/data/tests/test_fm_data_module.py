"""Tests for foundation model data modules (_fm_data_module.py)."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.data._fm_data_module import TTMDataModule, _TTMDataset
from pytorch_forecasting.data.timeseries import TimeSeries

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fm_timeseries():
    """TimeSeries with 1 target, 1 past-only covariate, 1 known-future covariate."""
    np.random.seed(0)
    n_series, n_timesteps = 20, 50
    rows = []
    for sid in range(n_series):
        for t in range(n_timesteps):
            rows.append(
                {
                    "series_id": sid,
                    "time_idx": t,
                    "target": np.sin(t / 5.0) + np.random.randn() * 0.1,
                    "feature_1": np.random.randn(),  # known-future continuous
                    "feature_2": np.random.randn(),  # past-only continuous
                }
            )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = TimeSeries(
            data=df,
            time="time_idx",
            target="target",
            group=["series_id"],
            num=["feature_1", "feature_2"],
            cat=[],
            known=["feature_1"],  # feature_1: known-future
            unknown=["feature_2"],  # feature_2: past-only
        )
    return ts


@pytest.fixture
def ttm_dm(fm_timeseries):
    """TTMDataModule fixture with small context/prediction lengths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = TTMDataModule(
            time_series_dataset=fm_timeseries,
            context_length=8,
            prediction_length=4,
            batch_size=4,
            num_workers=0,
        )
    return dm


# ---------------------------------------------------------------------------
# Task 2 tests:- skeleton wiring
# ---------------------------------------------------------------------------


def test_ttm_data_module_uses_ttm_dataset(ttm_dm):
    """TTMDataModule.setup() instantiates _TTMDataset for fit and predict stages."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("fit")
    assert isinstance(ttm_dm.train_dataset, _TTMDataset)
    assert isinstance(ttm_dm.val_dataset, _TTMDataset)

    # Also verify predict stage uses a fresh ttm_dm (function-scoped fixture)
    # so calling setup("predict") here is safe and does not affect other tests.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("predict")
    assert isinstance(ttm_dm.predict_dataset, _TTMDataset)


def test_ttm_dataset_captures_stage(ttm_dm):
    """_TTMDataset stores the stage at construction time."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("fit")
    assert ttm_dm.train_dataset._stage == "fit"


def test_channel_indices_derived(ttm_dm):
    """TTMDataModule.setup() derives prediction/exogenous channel indices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("fit")
    # 1 target -> prediction channels at index 0
    assert ttm_dm._prediction_channel_indices == [0]
    # 2 non-target continuous channels at indices 1 and 2 in past_values
    # (index 0 = target, 1 = feature_2 past-only, 2 = feature_1 known-future)
    assert ttm_dm._exogenous_channel_indices == [1, 2]


# ---------------------------------------------------------------------------
# Task 3 tests:- __getitem__ shapes and stage gating
# ---------------------------------------------------------------------------


def test_batch_shapes(ttm_dm):
    """past_values, past_observed_mask, and future_values have correct shapes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("fit")
    x, y = next(iter(ttm_dm.train_dataloader()))

    B = ttm_dm.batch_size
    ctx = ttm_dm.context_length  # 8
    pred = ttm_dm.prediction_length  # 4
    # Fixture: 1 target + 1 past-only + 1 known-future = 3 channels
    assert x["past_values"].shape == (B, ctx, 3)
    assert x["past_observed_mask"].shape == (B, ctx, 3)
    # Fixture has no NaN values, every position must be observed (all 1.0)
    assert x["past_observed_mask"].bool().all()
    # future_values: 1 known-future continuous covariate
    assert x["future_values"].shape == (B, pred, 1)
    # prediction_channel_indices: single list[int] after collate dedup
    assert x["prediction_channel_indices"] == [0]
    # y: single target -> 1-D per sample -> (B, pred) after stack
    assert y.shape == (B, pred)


def test_no_future_values_in_predict(ttm_dm):
    """future_values must not appear in x during the predict stage."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("predict")
    x, _ = next(iter(ttm_dm.predict_dataloader()))
    assert "future_values" not in x


# ---------------------------------------------------------------------------
# Task 4 test:- end-to-end forward pass with MockTTM
# ---------------------------------------------------------------------------


def test_mock_ttm_forward(ttm_dm):
    """
    A minimal mock model can consume a TTMDataModule batch without error.

    No tsfm_public import as MockTTM is a stand-in for the real TTM forward
    signature: forward(past_values, past_observed_mask=None, **kwargs).
    """
    pred_len = ttm_dm.prediction_length  # bound from fixture

    class MockTTM(nn.Module):
        def __init__(self, prediction_length):
            super().__init__()
            self._pred_len = prediction_length

        def forward(self, past_values, past_observed_mask=None, **kwargs):
            B, T, C = past_values.shape
            return {"prediction_outputs": torch.zeros(B, self._pred_len, 1)}

    model = MockTTM(pred_len)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ttm_dm.setup("fit")

    x, _ = next(iter(ttm_dm.train_dataloader()))

    out = model(
        x["past_values"],
        past_observed_mask=x["past_observed_mask"],
    )

    B = x["past_values"].shape[0]
    assert out["prediction_outputs"].shape == (B, pred_len, 1)
