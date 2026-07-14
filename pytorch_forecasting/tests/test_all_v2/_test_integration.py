from typing import Any

import torch

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.data import TimeSeries


def _integration(
    pkg: Base_pkg,
    test_data: dict[str, TimeSeries],
    datamodule_cfg: dict[str, Any],
    **kwargs,
):
    """Test integration of models with the `TimeSeries` and datamodules"""
    pkg.fit(test_data["train"])

    predictions = pkg.predict(
        test_data["predict"],
        mode="raw",
    )
    assert predictions is not None
    assert isinstance(predictions, dict)
    assert "prediction" in predictions

    pred_tensor = predictions["prediction"]
    assert isinstance(pred_tensor, torch.Tensor)
    assert pred_tensor.ndim == 3, f"Prediction must be 3D, got {pred_tensor.ndim}D"

    expected_pred_len = datamodule_cfg.get("prediction_length")
    if expected_pred_len:
        assert pred_tensor.shape[1] == expected_pred_len, (
            f"Pred length mismatch: expected {expected_pred_len}, "
            f"got {pred_tensor.shape[1]}"
        )
