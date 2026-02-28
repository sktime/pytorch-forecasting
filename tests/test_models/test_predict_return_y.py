"""Tests for predict with return_y=True and unequal batch sizes.

Regression tests for the bug where ``model.predict(dataloader, return_y=True)``
raises ``RuntimeError: Sizes of tensors must match except in dimension 1``
when the last batch is smaller than the rest, or when decoder lengths vary
across batches.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import Baseline, TimeSeriesDataSet


@pytest.fixture(scope="module")
def data():
    """Simple synthetic dataset for testing predict with return_y."""
    torch.manual_seed(23)
    np.random.seed(23)
    return pd.DataFrame(
        dict(
            value=np.random.rand(30) - 0.5,
            group=np.repeat(np.arange(3), 10),
            time_idx=np.tile(np.arange(10), 3),
        )
    )


@pytest.fixture(scope="module")
def dataset_fixed_length(data):
    """Dataset with fixed prediction length (min == max)."""
    return TimeSeriesDataSet(
        data,
        group_ids=["group"],
        target="value",
        time_idx="time_idx",
        max_encoder_length=2,
        max_prediction_length=1,
        time_varying_unknown_reals=["value"],
    )


@pytest.fixture(scope="module")
def dataset_variable_length(data):
    """Dataset with variable prediction length (min < max).

    This triggers per-batch padding via ``rnn.pad_sequence`` in the collate
    function, which can lead to different ``dim=1`` across batches.
    """
    return TimeSeriesDataSet(
        data,
        group_ids=["group"],
        target="value",
        time_idx="time_idx",
        max_encoder_length=4,
        min_encoder_length=2,
        max_prediction_length=3,
        min_prediction_length=1,
        time_varying_unknown_reals=["value"],
    )


def test_predict_return_y_unequal_last_batch(dataset_fixed_length):
    """Test predict(return_y=True) when last batch is smaller than batch_size."""
    n_samples = len(dataset_fixed_length)
    # pick a batch_size that does not evenly divide the dataset
    batch_size = max(2, n_samples // 3 + 1)
    dataloader = dataset_fixed_length.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    result = Baseline().predict(dataloader, return_y=True)

    assert result.y is not None
    y, weight = result.y
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == n_samples


def test_predict_return_y_variable_decoder_length(dataset_variable_length):
    """Test predict(return_y=True) with variable decoder lengths across batches.

    When ``min_prediction_length < max_prediction_length``, different batches
    may have different maximum decoder lengths.  The collate function pads
    targets per-batch via ``rnn.pad_sequence``, so the ``dim=1`` of the target
    tensor can differ across batches.  Previously this caused a RuntimeError
    in ``concat_sequences`` which used bare ``torch.cat(dim=0)``.
    """
    n_samples = len(dataset_variable_length)
    batch_size = max(2, n_samples // 3 + 1)
    dataloader = dataset_variable_length.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    result = Baseline().predict(dataloader, return_y=True)

    assert result.y is not None
    y, weight = result.y
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == n_samples


def test_predict_return_y_batch_size_one(dataset_fixed_length):
    """Test predict(return_y=True) with batch_size=1 (every batch is size 1)."""
    dataloader = dataset_fixed_length.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    result = Baseline().predict(dataloader, return_y=True)

    assert result.y is not None
    y, weight = result.y
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == len(dataset_fixed_length)


def test_predict_return_y_false_still_works(dataset_fixed_length):
    """Ensure return_y=False still works after the fix (no regression)."""
    n_samples = len(dataset_fixed_length)
    batch_size = max(2, n_samples // 3 + 1)
    dataloader = dataset_fixed_length.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    result = Baseline().predict(dataloader, return_y=False)

    # When return_y=False the result is a plain tensor, not a Prediction tuple
    assert isinstance(result, torch.Tensor)
