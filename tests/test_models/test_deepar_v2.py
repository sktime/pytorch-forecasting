from typing import Literal

import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.metrics import MAE, SMAPE, NormalDistributionLoss
from pytorch_forecasting.models.deepar._deepar_v2 import DeepAR as DeepAR_v2

BATCH_SIZE_TEST = 2
MAX_ENCODER_LENGTH_TEST = 10
MAX_PREDICTION_LENGTH_TEST = 5
HIDDEN_SIZE_TEST = 8
RNN_LAYERS_TEST = 1
DROPOUT_TEST = 0.1


def get_default_test_metadata(
    enc_cont=2,
    enc_cat=1,
    dec_cont=2,
    dec_cat=1,
    target_dim=1,
):
    return {
        "max_encoder_length": MAX_ENCODER_LENGTH_TEST,
        "max_prediction_length": MAX_PREDICTION_LENGTH_TEST,
        "encoder_cont": enc_cont,
        "encoder_cat": enc_cat,
        "decoder_cont": dec_cont,
        "decoder_cat": dec_cat,
        "target_dim": target_dim,
    }


def create_deepar_input_batch_for_test(
    metadata, batch_size=BATCH_SIZE_TEST, device="cpu"
):
    """Create a synthetic input batch dictionary for testing DeepAR forward passes."""
    x = {
        "encoder_cont": torch.randn(
            batch_size,
            metadata["max_encoder_length"],
            metadata.get("encoder_cont", 0),
            device=device,
        ),
        "encoder_cat": torch.randn(
            batch_size,
            metadata["max_encoder_length"],
            metadata.get("encoder_cat", 0),
            device=device,
        ),
        "decoder_cont": torch.randn(
            batch_size,
            metadata["max_prediction_length"],
            metadata.get("decoder_cont", 0),
            device=device,
        ),
        "decoder_cat": torch.randn(
            batch_size,
            metadata["max_prediction_length"],
            metadata.get("decoder_cat", 0),
            device=device,
        ),
        "encoder_lengths": torch.full(
            (batch_size,),
            metadata["max_encoder_length"],
            dtype=torch.long,
            device=device,
        ),
        "decoder_lengths": torch.full(
            (batch_size,),
            metadata["max_prediction_length"],
            dtype=torch.long,
            device=device,
        ),
    }
    return x


@pytest.fixture
def deepar_model_params_fixture():
    """Create basic model parameters for DeepAR."""
    return {
        "loss": NormalDistributionLoss(),
        "hidden_size": HIDDEN_SIZE_TEST,
        "rnn_layers": RNN_LAYERS_TEST,
        "dropout": DROPOUT_TEST,
        "cell_type": "LSTM",
    }


def test_deepar_v2_initialization(deepar_model_params_fixture):
    """Test basic initialization of the DeepAR V2 model."""
    metadata = get_default_test_metadata()
    model = DeepAR_v2(**deepar_model_params_fixture, metadata=metadata)

    assert model.hidden_size == HIDDEN_SIZE_TEST
    assert model.rnn_layers == RNN_LAYERS_TEST
    assert model.max_encoder_length == MAX_ENCODER_LENGTH_TEST
    assert model.max_prediction_length == MAX_PREDICTION_LENGTH_TEST
    assert model.encoder_cont_dim == metadata["encoder_cont"]
    assert model.target_dim == metadata["target_dim"]


@pytest.mark.parametrize("cell_type", ["LSTM", "GRU"])
def test_deepar_v2_forward_pass(deepar_model_params_fixture, cell_type):
    """Test DeepAR V2 forward pass with different cell types."""
    metadata = get_default_test_metadata()
    params = deepar_model_params_fixture.copy()
    params["cell_type"] = cell_type

    model = DeepAR_v2(**params, metadata=metadata)
    model.eval()

    x = create_deepar_input_batch_for_test(metadata)
    output = model(x)

    assert "prediction" in output
    prediction = output["prediction"]

    n_dist_params = len(params["loss"].distribution_arguments)
    assert prediction.shape == (
        BATCH_SIZE_TEST,
        MAX_PREDICTION_LENGTH_TEST,
        metadata["target_dim"] * n_dist_params,
    )
    assert not torch.isnan(prediction).any()


def test_deepar_v2_multi_target(deepar_model_params_fixture):
    """Test DeepAR V2 forward pass with multiple targets."""
    target_dim = 3
    metadata = get_default_test_metadata(target_dim=target_dim)
    model = DeepAR_v2(**deepar_model_params_fixture, metadata=metadata)
    model.eval()

    x = create_deepar_input_batch_for_test(metadata)
    output = model(x)

    prediction = output["prediction"]
    n_dist_params = len(deepar_model_params_fixture["loss"].distribution_arguments)

    if isinstance(prediction, list):
        assert len(prediction) == target_dim
        for p in prediction:
            assert p.shape == (
                BATCH_SIZE_TEST,
                MAX_PREDICTION_LENGTH_TEST,
                n_dist_params,
            )
    else:
        assert prediction.shape == (
            BATCH_SIZE_TEST,
            MAX_PREDICTION_LENGTH_TEST,
            target_dim,
            n_dist_params,
        )
