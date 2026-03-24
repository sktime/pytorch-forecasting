"""
Minimal validation tests for model and package templates.
"""

import pytest
import torch

from pytorch_forecasting.models.model_template import MyNewModel
from pytorch_forecasting.models.pkg_template import MyNewModel_pkg


def test_template_registry_discovery():
    """
    Verify model package discovery.
    """
    # 1. Test get_cls()
    model_cls = MyNewModel_pkg.get_cls()
    assert model_cls == MyNewModel

    # 2. Test tags
    assert "info:name" in MyNewModel_pkg._tags
    assert MyNewModel_pkg._tags["info:name"] == "MyNewModel"


def test_template_forward_pass():
    """
    Verify basic forward pass handling.
    """
    metadata = {
        "encoder_cont": 1,
        "encoder_cat": 0,
        "max_prediction_length": 1,
        "target": 1,
        "max_encoder_length": 5,
    }

    # Instantiate the model
    model = MyNewModel(metadata=metadata, loss=torch.nn.MSELoss(), hidden_size=16)

    # Prepare a dummy batch
    x = {"encoder_cont": torch.randn(1, 5, 1)}

    # Expect NotImplementedError for template
    with pytest.raises(NotImplementedError, match="Implement forward logic."):
        model(x)


def test_template_training_step():
    """
    Verify training_step accessibility.
    """
    metadata = {
        "encoder_cont": 1,
        "encoder_cat": 0,
        "max_prediction_length": 1,
        "target": 1,
        "max_encoder_length": 5,
    }
    model = MyNewModel(metadata=metadata, loss=torch.nn.MSELoss(), hidden_size=16)

    batch = ({"encoder_cont": torch.randn(1, 5, 1)}, torch.randn(1, 1, 1))

    try:
        model.training_step(batch, 0)
    except Exception:  # noqa: S110
        pass


def test_template_pkg_metadata():
    """
    Verify package metadata methods.
    """
    MyNewModel_pkg.get_datamodule_cls()
    params = MyNewModel_pkg.get_test_train_params()
    assert isinstance(params, list)
    assert len(params) > 0
