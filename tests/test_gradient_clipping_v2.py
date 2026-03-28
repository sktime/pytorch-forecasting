import pytest
import torch
from unittest.mock import MagicMock
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer
from pytorch_forecasting.models.samformer._samformer_v2 import Samformer
from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

class MockMetric(torch.nn.Module):
    def forward(self, y_hat, y):
        return torch.tensor(0.0)
    def to_prediction(self, out):
        return out
    def to_quantiles(self, out):
        return out

def test_base_model_gradient_clipping_params():
    model = BaseModel(loss=MockMetric(), gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    assert model.gradient_clip_val == 0.5
    assert model.gradient_clip_algorithm == "norm"

def test_configure_gradient_clipping_priority():
    model = BaseModel(loss=MockMetric(), gradient_clip_val=0.5, gradient_clip_algorithm="value")
    model.clip_gradients = MagicMock()
    optimizer = MagicMock()
    
    # Priority should be given to model's gradient_clip_val
    model.configure_gradient_clipping(optimizer, gradient_clip_val=0.1, gradient_clip_algorithm="norm")
    model.clip_gradients.assert_called_once_with(
        optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="value"
    )

def test_sensitive_models_defaults():
    metadata = {
        "context_length": 10,
        "max_encoder_length": 10,
        "encoder_cont": 1,
        "encoder_cat": 0,
        "max_prediction_length": 1,
        "target": 1,
        "decoder_cont": 0,
        "decoder_cat": 0,
    }
    
    # TimeXer
    timexer = TimeXer(loss=MockMetric(), metadata=metadata)
    assert timexer.gradient_clip_val == 0.1
    assert timexer.sensitive_to_gradient_explosions is True
    
    # Samformer
    samformer = Samformer(loss=MockMetric(), hidden_size=16, use_revin=True, metadata=metadata)
    assert samformer.gradient_clip_val == 0.1
    assert samformer.sensitive_to_gradient_explosions is True
    
    # TFT
    tft = TFT(loss=MockMetric(), metadata=metadata)
    assert tft.gradient_clip_val == 0.1
    assert tft.sensitive_to_gradient_explosions is True

def test_sensitive_model_warning():
    model = BaseModel(loss=MockMetric())
    model.sensitive_to_gradient_explosions = True
    model.gradient_clip_val = None
    
    optimizer = MagicMock()
    with pytest.warns(UserWarning, match="is sensitive to gradient explosions"):
        model.configure_gradient_clipping(optimizer)

if __name__ == "__main__":
    pytest.main([__file__])
