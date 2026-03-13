import torch
from torch import nn

from pytorch_forecasting.models.foundation import FoundationModelWrapper


def test_foundation_wrapper_forward():
    batch_size = 4
    encoder_length = 6
    n_features = 2
    output_size = 3

    pretrained = nn.Linear(n_features, output_size)

    wrapper = FoundationModelWrapper(pretrained_model=pretrained)

    batch = {
        "encoder_cont": torch.randn(batch_size, encoder_length, n_features),
    }

    out = wrapper(batch)

    assert hasattr(out, "prediction")
    assert out.prediction.shape == (batch_size, encoder_length, output_size)
