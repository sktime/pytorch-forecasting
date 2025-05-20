import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

BATCH_SIZE_TEST = 2
MAX_ENCODER_LENGTH_TEST = 10
MAX_PREDICTION_LENGTH_TEST = 5
HIDDEN_SIZE_TEST = 8
OUTPUT_SIZE_TEST = 1
ATTENTION_HEAD_SIZE_TEST = 2
NUM_LAYERS_TEST = 1
DROPOUT_TEST = 0.1


def get_default_test_metadata(
    enc_cont=2,
    enc_cat=1,
    dec_cont=1,
    dec_cat=1,
    static_cat=1,
    static_cont=1,
    output_size=OUTPUT_SIZE_TEST,
):
    """Return a dict representing default metadata for TFT model initialization."""
    return {
        "max_encoder_length": MAX_ENCODER_LENGTH_TEST,
        "max_prediction_length": MAX_PREDICTION_LENGTH_TEST,
        "encoder_cont": enc_cont,
        "encoder_cat": enc_cat,
        "decoder_cont": dec_cont,
        "decoder_cat": dec_cat,
        "static_categorical_features": static_cat,
        "static_continuous_features": static_cont,
        "target": output_size,
    }


def create_tft_input_batch_for_test(metadata, batch_size=BATCH_SIZE_TEST, device="cpu"):
    """Create a synthetic input batch dictionary for testing TFT forward passes."""

    def _get_dim_val(key):
        return metadata.get(key, 0)

    x = {
        "encoder_cont": torch.randn(
            batch_size,
            metadata["max_encoder_length"],
            _get_dim_val("encoder_cont"),
            device=device,
        ),
        "encoder_cat": torch.randn(
            batch_size,
            metadata["max_encoder_length"],
            _get_dim_val("encoder_cat"),
            device=device,
        ),
        "decoder_cont": torch.randn(
            batch_size,
            metadata["max_prediction_length"],
            _get_dim_val("decoder_cont"),
            device=device,
        ),
        "decoder_cat": torch.randn(
            batch_size,
            metadata["max_prediction_length"],
            _get_dim_val("decoder_cat"),
            device=device,
        ),
        "static_categorical_features": torch.randn(
            batch_size, 1, _get_dim_val("static_categorical_features"), device=device
        ),
        "static_continuous_features": torch.randn(
            batch_size, 1, _get_dim_val("static_continuous_features"), device=device
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
        "groups": torch.arange(batch_size, device=device).unsqueeze(1),
        "encoder_time_idx": torch.stack(
            [torch.arange(metadata["max_encoder_length"], device=device)] * batch_size
        ),
        "decoder_time_idx": torch.stack(
            [
                torch.arange(
                    metadata["max_encoder_length"],
                    metadata["max_encoder_length"] + metadata["max_prediction_length"],
                    device=device,
                )
            ]
            * batch_size
        ),
        "target_scale": torch.ones((batch_size, 1), device=device),
    }
    return x


dummy_loss_for_test = nn.MSELoss()


@pytest.fixture(scope="module")
def tft_model_params_fixture_func():
    """Create a default set of model parameters for TFT."""
    return {
        "loss": dummy_loss_for_test,
        "hidden_size": HIDDEN_SIZE_TEST,
        "num_layers": NUM_LAYERS_TEST,
        "attention_head_size": ATTENTION_HEAD_SIZE_TEST,
        "dropout": DROPOUT_TEST,
        "output_size": OUTPUT_SIZE_TEST,
    }


def test_basic_initialization(tft_model_params_fixture_func):
    """Test basic initialization of the TFT model with default metadata.

    Verifies:
    - Model attributes match the provided metadata (e.g., hidden_size, num_layers).
    - Proper construction of key model components (LSTM, attention, etc.).
    - Correct dimensionality of input layers based on metadata.
    - Model retains metadata and hyperparameters as expected.
    """
    metadata = get_default_test_metadata(output_size=OUTPUT_SIZE_TEST)
    model = TFT(**tft_model_params_fixture_func, metadata=metadata)
    assert model.hidden_size == HIDDEN_SIZE_TEST
    assert model.num_layers == NUM_LAYERS_TEST
    assert hasattr(model, "metadata") and model.metadata == metadata
    assert model.encoder_input_dim == metadata["encoder_cont"] + metadata["encoder_cat"]
    assert (
        model.static_input_dim
        == metadata["static_categorical_features"]
        + metadata["static_continuous_features"]
    )
    assert isinstance(model.lstm_encoder, nn.LSTM)
    assert model.lstm_encoder.input_size == max(1, model.encoder_input_dim)
    assert isinstance(model.self_attention, nn.MultiheadAttention)
    if hasattr(model, "hparams") and model.hparams:
        assert model.hparams.get("hidden_size") == HIDDEN_SIZE_TEST
    assert model.output_size == OUTPUT_SIZE_TEST


def test_initialization_no_time_varying_features(tft_model_params_fixture_func):
    """Test TFT initialization with no time-varying (encoder/decoder) features.

    Verifies:
    - Model handles zero encoder/decoder input dimensions correctly.
    - Skips creation of encoder/decoder variable selection networks.
    - Defaults to input size 1 for LSTMs when no time-varying features exist.
    """
    metadata = get_default_test_metadata(
        enc_cont=0, enc_cat=0, dec_cont=0, dec_cat=0, output_size=OUTPUT_SIZE_TEST
    )
    model = TFT(**tft_model_params_fixture_func, metadata=metadata)
    assert model.encoder_input_dim == 0
    assert model.encoder_var_selection is None
    assert model.lstm_encoder.input_size == 1
    assert model.decoder_input_dim == 0
    assert model.decoder_var_selection is None
    assert model.lstm_decoder.input_size == 1


def test_initialization_no_static_features(tft_model_params_fixture_func):
    """Test TFT initialization with no static features.

    Verifies:
    - Model static input dim is 0.
    - Static context linear layer is not created.
    """
    metadata = get_default_test_metadata(
        static_cat=0, static_cont=0, output_size=OUTPUT_SIZE_TEST
    )
    model = TFT(**tft_model_params_fixture_func, metadata=metadata)
    assert model.static_input_dim == 0
    assert model.static_context_linear is None


@pytest.mark.parametrize(
    "enc_c, enc_k, dec_c, dec_k, stat_c, stat_k",
    [
        (2, 1, 1, 1, 1, 1),
        (2, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 1, 0),
        (1, 0, 1, 0, 0, 1),
    ],
)
def test_forward_pass_configs(
    tft_model_params_fixture_func, enc_c, enc_k, dec_c, dec_k, stat_c, stat_k
):
    """Test TFT forward pass across multiple feature configurations.

    Verifies:
    - Model can forward pass without errors for varying combinations of input types.
    - Output prediction tensor has expected shape.
    - Output contains no NaNs or infinities.
    """
    current_tft_actual_output_size = tft_model_params_fixture_func["output_size"]
    metadata = get_default_test_metadata(
        enc_cont=enc_c,
        enc_cat=enc_k,
        dec_cont=dec_c,
        dec_cat=dec_k,
        static_cat=stat_c,
        static_cont=stat_k,
        output_size=current_tft_actual_output_size,
    )
    model_params = tft_model_params_fixture_func.copy()
    model_params["output_size"] = current_tft_actual_output_size
    model = TFT(**model_params, metadata=metadata)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = create_tft_input_batch_for_test(
        metadata, batch_size=BATCH_SIZE_TEST, device=device
    )
    output_dict = model(x)
    predictions = output_dict["prediction"]
    assert predictions.shape == (
        BATCH_SIZE_TEST,
        MAX_PREDICTION_LENGTH_TEST,
        current_tft_actual_output_size,
    )
    assert not torch.isnan(predictions).any(), "NaNs in prediction"
    assert not torch.isinf(predictions).any(), "Infs in prediction"


@pytest.fixture
def sample_pandas_data_for_test():
    """Create synthetic multivariate time series data as a pandas DataFrame."""
    series_len = MAX_ENCODER_LENGTH_TEST + MAX_PREDICTION_LENGTH_TEST + 5
    num_groups = 6
    data = []

    for i in range(num_groups):
        static_cont_val = np.float32(i * 10.0)
        static_cat_code = np.float32(i % 2)

        df_group = pd.DataFrame(
            {
                "time_idx": np.arange(series_len, dtype=np.int64),
                "group_id_str": np.repeat(f"g{i}", series_len),
                "target": np.random.rand(series_len).astype(np.float32) + i,
                "enc_cont1": np.random.rand(series_len).astype(np.float32),
                "enc_cat1_codes": np.random.randint(0, 3, series_len).astype(
                    np.float32
                ),
                "dec_known_cont": np.sin(np.arange(series_len) / 5.0).astype(
                    np.float32
                ),
                "dec_known_cat_codes": np.random.randint(0, 2, series_len).astype(
                    np.float32
                ),
                "static_cat_feat_codes": np.full(
                    series_len, static_cat_code, dtype=np.float32
                ),
                "static_cont_feat": np.full(
                    series_len, static_cont_val, dtype=np.float32
                ),
            }
        )
        data.append(df_group)

    df = pd.concat(data, ignore_index=True)

    df["group_id"] = df["group_id_str"].astype("category")
    df.drop(columns=["group_id_str"], inplace=True)

    return df


@pytest.fixture
def timeseries_obj_for_test(sample_pandas_data_for_test):
    """Convert sample DataFrame into a TimeSeries object."""
    df = sample_pandas_data_for_test

    return TimeSeries(
        data=df,
        time="time_idx",
        target="target",
        group=["group_id"],
        num=[
            "enc_cont1",
            "enc_cat1_codes",
            "dec_known_cont",
            "dec_known_cat_codes",
            "static_cat_feat_codes",
            "static_cont_feat",
        ],
        cat=[],
        known=["dec_known_cont", "dec_known_cat_codes", "time_idx"],
        static=["static_cat_feat_codes", "static_cont_feat"],
    )


@pytest.fixture
def data_module_for_test(timeseries_obj_for_test):
    """Initialize and sets up an EncoderDecoderTimeSeriesDataModule."""
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=timeseries_obj_for_test,
        batch_size=BATCH_SIZE_TEST,
        max_encoder_length=MAX_ENCODER_LENGTH_TEST,
        max_prediction_length=MAX_PREDICTION_LENGTH_TEST,
        train_val_test_split=(0.5, 0.25, 0.25),
    )
    dm.setup("fit")
    dm.setup("test")
    return dm


def test_model_with_datamodule_integration(
    tft_model_params_fixture_func, data_module_for_test
):
    """Integration test to ensure TFT works correctly with data module.

    Verifies:
    - Metadata inferred from data module matches expected input dimensions.
    - Model processes real dataloader batches correctly.
    - Output and target tensors from model and data module align in shape.
    - No NaNs in predictions.
    """
    dm = data_module_for_test
    model_metadata_from_dm = dm.metadata

    assert (
        model_metadata_from_dm["encoder_cont"] == 6
    ), f"Actual encoder_cont: {model_metadata_from_dm['encoder_cont']}"
    assert (
        model_metadata_from_dm["encoder_cat"] == 0
    ), f"Actual encoder_cat: {model_metadata_from_dm['encoder_cat']}"
    assert (
        model_metadata_from_dm["decoder_cont"] == 2
    ), f"Actual decoder_cont: {model_metadata_from_dm['decoder_cont']}"
    assert (
        model_metadata_from_dm["decoder_cat"] == 0
    ), f"Actual decoder_cat: {model_metadata_from_dm['decoder_cat']}"
    assert (
        model_metadata_from_dm["static_categorical_features"] == 0
    ), f"Actual static_cat: {model_metadata_from_dm['static_categorical_features']}"
    assert (
        model_metadata_from_dm["static_continuous_features"] == 2
    ), f"Actual static_cont: {model_metadata_from_dm['static_continuous_features']}"
    assert model_metadata_from_dm["target"] == 1

    tft_init_args = tft_model_params_fixture_func.copy()
    tft_init_args["output_size"] = model_metadata_from_dm["target"]
    model = TFT(**tft_init_args, metadata=model_metadata_from_dm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    train_loader = dm.train_dataloader()
    batch_x, batch_y = next(iter(train_loader))

    actual_batch_size = batch_x["encoder_cont"].shape[0]
    batch_x = {k: v.to(device) for k, v in batch_x.items()}
    batch_y = batch_y.to(device)

    assert batch_x["encoder_cont"].shape[2] == model_metadata_from_dm["encoder_cont"]
    assert batch_x["encoder_cat"].shape[2] == model_metadata_from_dm["encoder_cat"]
    assert batch_x["decoder_cont"].shape[2] == model_metadata_from_dm["decoder_cont"]
    assert batch_x["decoder_cat"].shape[2] == model_metadata_from_dm["decoder_cat"]
    assert (
        batch_x["static_categorical_features"].shape[2]
        == model_metadata_from_dm["static_categorical_features"]
    )
    assert (
        batch_x["static_continuous_features"].shape[2]
        == model_metadata_from_dm["static_continuous_features"]
    )

    output_dict = model(batch_x)
    predictions = output_dict["prediction"]
    assert predictions.shape == (
        actual_batch_size,
        MAX_PREDICTION_LENGTH_TEST,
        model_metadata_from_dm["target"],
    )
    assert not torch.isnan(predictions).any()
    assert batch_y.shape == (
        actual_batch_size,
        MAX_PREDICTION_LENGTH_TEST,
        model_metadata_from_dm["target"],
    )
