import pytest
import torch.nn as nn

from pytorch_forecasting.models.samformer._samformer_v2 import Samformer


def test_samformer_metadata_validation():
    """Verify that Samformer raises ValueError when initialized with metadata=None."""
    with pytest.raises(ValueError, match="Samformer requires 'metadata' dictionary"):
        Samformer(loss=nn.MSELoss(), hidden_size=512, use_revin=True, metadata=None)
