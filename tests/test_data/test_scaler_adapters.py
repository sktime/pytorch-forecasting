import torch

from pytorch_forecasting.adapters.scaler_adapters import ScalerAdapter
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    MultiNormalizer,
    TorchNormalizer,
)


def test_sequence_transform_leaves_global_normalizer_data_unchanged():
    """Test that sequence transforms do not reapply global normalization."""
    data = torch.tensor([1.0, 2.0, 3.0])
    adapter = ScalerAdapter(TorchNormalizer())

    transformed = adapter.transform_sequence(data)

    torch.testing.assert_close(transformed, data)


def test_sequence_transform_supports_one_dimensional_multi_normalizer_data():
    """Test one-dimensional data with a single-target MultiNormalizer."""
    encoder_data = torch.tensor([1.0, 2.0, 3.0])
    decoder_data = torch.tensor([4.0, 5.0])
    adapter = ScalerAdapter(MultiNormalizer([EncoderNormalizer()]))

    transformed_encoder = adapter.fit_transform_sequence(encoder_data)
    transformed_decoder = adapter.transform_sequence(decoder_data)

    scale = encoder_data.std() + torch.finfo(encoder_data.dtype).eps
    expected_encoder = (encoder_data - encoder_data.mean()) / scale
    expected_decoder = (decoder_data - encoder_data.mean()) / scale

    assert transformed_encoder.shape == (3, 1)
    assert transformed_decoder.shape == (2, 1)
    torch.testing.assert_close(transformed_encoder.squeeze(-1), expected_encoder)
    torch.testing.assert_close(transformed_decoder.squeeze(-1), expected_decoder)
