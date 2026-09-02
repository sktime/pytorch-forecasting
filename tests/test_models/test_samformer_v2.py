import pytest
import torch.nn as nn

from pytorch_forecasting.models.samformer._samformer_v2 import Samformer


def test_samformer_requires_metadata():
    """Test that Samformer rejects missing metadata explicitly."""
    with pytest.raises(ValueError, match="metadata is required"):
        Samformer(loss=nn.MSELoss(), hidden_size=512, use_revin=True, metadata=None)
