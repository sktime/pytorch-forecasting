import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.layers import (
    AttentionLayer,
    FullAttention,
    iTransformerEncoder,
    iTransformerEncoderLayer,
)
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.itransformer import (
    iTransformer,
    iTransformerForecaster,
)


def _make_metadata(
    context_length: int = 12,
    prediction_length: int = 4,
    cont_dim: int = 2,
    target_dim: int = 1,
):
    return {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "n_features": {
            "continuous": cont_dim,
            "categorical": 0,
            "target": target_dim,
        },
        "feature_indices": {
            "continuous": list(range(cont_dim)),
            "categorical": [],
            "known": [],
            "unknown": [],
            "target": list(range(cont_dim, cont_dim + target_dim)),
        },
    }


def _make_batch(
    batch_size: int = 4,
    context_length: int = 12,
    cont_dim: int = 2,
    target_dim: int = 1,
):
    batch = {
        "history_target": torch.randn(batch_size, context_length, target_dim),
    }
    if cont_dim > 0:
        batch["history_cont"] = torch.randn(batch_size, context_length, cont_dim)
    return batch


@pytest.mark.parametrize("use_norm", [True, False])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
@pytest.mark.parametrize("cont_dim", [0, 2])
def test_itransformer_forward_point(use_norm, activation, cont_dim):
    metadata = _make_metadata(cont_dim=cont_dim, target_dim=1)
    model = iTransformer(
        loss=MAE(),
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        activation=activation,
        use_norm=use_norm,
        metadata=metadata,
    )
    batch = _make_batch(
        batch_size=4, context_length=12, cont_dim=cont_dim, target_dim=1
    )
    out = model(batch)

    assert "prediction" in out
    assert out["prediction"].shape == (4, 4, 1)
    assert torch.isfinite(out["prediction"]).all()


def test_itransformer_forward_multivariate():
    metadata = _make_metadata(cont_dim=2, target_dim=3)
    model = iTransformer(
        loss=nn.MSELoss(),
        hidden_size=16,
        n_heads=2,
        e_layers=2,
        d_ff=32,
        metadata=metadata,
    )
    batch = _make_batch(batch_size=4, context_length=12, cont_dim=2, target_dim=3)
    out = model(batch)

    assert out["prediction"].shape == (4, 4, 3)
    assert torch.isfinite(out["prediction"]).all()


def test_itransformer_forward_quantile():
    quantiles = [0.1, 0.5, 0.9]
    metadata = _make_metadata(cont_dim=2, target_dim=1)
    model = iTransformer(
        loss=QuantileLoss(quantiles=quantiles),
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        metadata=metadata,
    )
    batch = _make_batch(batch_size=4, context_length=12, cont_dim=2, target_dim=1)
    out = model(batch)

    assert out["prediction"].shape == (4, 4, len(quantiles))
    assert torch.isfinite(out["prediction"]).all()


def test_itransformer_encoder_layer():
    layer = iTransformerEncoderLayer(
        attention=AttentionLayer(
            FullAttention(
                False, factor=1, attention_dropout=0.0, output_attention=False
            ),
            d_model=16,
            n_heads=2,
        ),
        d_model=16,
        d_ff=32,
        dropout=0.0,
        activation="gelu",
    )
    encoder = iTransformerEncoder([layer], norm_layer=nn.LayerNorm(16))
    x = torch.randn(4, 5, 16)
    out, attns = encoder(x)

    assert out.shape == (4, 5, 16)
    assert len(attns) == 1


def test_itransformer_forecaster_pkg_binding():
    assert iTransformer._pkg() is iTransformerForecaster
    assert iTransformerForecaster.get_cls() is iTransformer


def test_itransformer_invalid_heads():
    metadata = _make_metadata(cont_dim=2, target_dim=1)
    with pytest.raises(ValueError, match="divisible"):
        iTransformer(loss=MAE(), hidden_size=15, n_heads=4, metadata=metadata)


def test_itransformer_forecaster_end_to_end():
    from lightning.pytorch import Trainer

    from pytorch_forecasting.data.data_module import TslibDataModule
    from pytorch_forecasting.data.examples import load_toydata
    from pytorch_forecasting.data.timeseries import TimeSeries

    df = load_toydata(num_series=10, seq_length=25)
    data = TimeSeries(
        df[["series_id", "time_idx", "x", "future_known_feature", "y"]],
        time="time_idx",
        target="y",
        group=["series_id"],
        known=["future_known_feature"],
        unknown=["x"],
    )
    datamodule = TslibDataModule(
        context_length=8,
        prediction_length=2,
        batch_size=4,
    )
    forecaster = iTransformerForecaster(
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        use_efficient_attention=True,
        trainer=Trainer(max_epochs=1, enable_checkpointing=False, logger=False),
        datamodule=datamodule,
    )
    forecaster.fit(data)
    preds = forecaster.predict(data)
    assert preds is not None
