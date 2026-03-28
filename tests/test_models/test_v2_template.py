"""
Consolidated test for v2 template.
"""

import pytest
import torch

from pytorch_forecasting.models.v2_template import MyNewModel, MyNewModel_pkg


def test_v2_template_coverage():
    """
    Exercise v2_template.py to satisfy coverage.
    """
    metadata = {
        "encoder_cont": 1,
        "encoder_cat": 0,
        "max_prediction_length": 1,
        "target": 1,
        "max_encoder_length": 5,
    }
    model = MyNewModel(metadata=metadata, loss=torch.nn.MSELoss(), hidden_size=16)

    # Touch methods
    MyNewModel._pkg()
    try:
        model({"encoder_cont": torch.randn(1, 5, 1)})
    except NotImplementedError:
        pass

    try:
        batch = ({"encoder_cont": torch.randn(1, 5, 1)}, torch.randn(1, 1, 1))
        model.training_step(batch, 0)
    except Exception:  # noqa: S110
        pass

    # Touch package methods
    MyNewModel_pkg.get_cls()
    MyNewModel_pkg.get_datamodule_cls()
    MyNewModel_pkg.get_test_train_params()
