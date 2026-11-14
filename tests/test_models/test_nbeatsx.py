"""NBEATSx integration and covariate routing tests."""

import copy

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import NBeatsX, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.layers._nbeats._nbeatsx import NBEATSxBlock
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.nbeats import NBeatsX_pkg


@pytest.fixture
def dataset():
    time = np.tile(np.arange(30), 2)
    group = np.repeat(["a", "b"], 30)
    data = pd.DataFrame(
        {
            "time": time,
            "group": group,
            "target": np.sin(time / 3) + 0.1 * time,
            "known": np.cos(time / 3),
            "observed": np.sin(time / 2),
            "static": np.repeat([1.0, 2.0], 30),
            "calendar": (time % 3).astype(str),
            "event": (time % 2).astype(str),
            "observed_category": (time % 2).astype(str),
        }
    )
    return TimeSeriesDataSet(
        data,
        time_idx="time",
        target="target",
        group_ids=["group"],
        max_encoder_length=6,
        max_prediction_length=3,
        static_categoricals=["group"],
        static_reals=["static"],
        time_varying_known_reals=["known"],
        time_varying_unknown_reals=["observed", "target"],
        time_varying_known_categoricals=["calendar", "event"],
        time_varying_unknown_categoricals=["observed_category"],
        target_normalizer=TorchNormalizer(method="standard"),
    )


def make_model(dataset, **kwargs):
    return NBeatsX.from_dataset(
        dataset,
        widths=[8],
        num_block_layers=[2],
        dropout=0,
        loss=MAE(),
        **kwargs,
    )


@pytest.mark.parametrize(
    "stack_types", [["generic"], ["trend", "seasonality", "exogenous"]]
)
def test_covariates_and_gradients(dataset, stack_types):
    torch.manual_seed(42)
    model = make_model(dataset, stack_types=stack_types, num_blocks=[1])
    x, y = next(iter(dataset.to_dataloader(train=False, batch_size=4)))
    x["encoder_cont"].requires_grad_()
    x["decoder_cont"].requires_grad_()
    output = model(x)
    assert output.prediction.shape == (4, 3, 1)
    assert output.backcast.shape == (4, 6, 1)
    assert output.block_forecasts.shape == (4, 3, len(stack_types))
    torch.testing.assert_close(
        model.transform_output(
            output.block_forecasts.sum(-1).unsqueeze(-1), x["target_scale"]
        ),
        output.prediction,
    )
    torch.testing.assert_close(
        model.transform_output(
            output.block_backcasts.sum(-1).unsqueeze(-1), x["target_scale"]
        ),
        output.backcast,
    )
    loss = model.loss(output.prediction, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    for name in ["target", "observed", "known", "static"]:
        assert x["encoder_cont"].grad[..., dataset.reals.index(name)].abs().sum() > 0
    assert x["decoder_cont"].grad[..., dataset.reals.index("known")].abs().sum() > 0
    for name in ["target", "observed", "static"]:
        assert (
            torch.count_nonzero(x["decoder_cont"].grad[..., dataset.reals.index(name)])
            == 0
        )
    for embedding in model.embeddings.values():
        assert embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "stack_types", [["generic"], ["trend", "seasonality", "exogenous"]]
)
def test_no_future_leakage(dataset, stack_types):
    model = make_model(dataset, stack_types=stack_types, num_blocks=[1]).eval()
    x, _ = next(iter(dataset.to_dataloader(train=False, batch_size=4)))
    expected = model(x).prediction
    changed = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in x.items()
    }
    for name in ["target", "observed", "static"]:
        changed["decoder_cont"][..., dataset.reals.index(name)] = float("nan")
    for name in ["observed_category", "group"]:
        changed["decoder_cat"][..., dataset.flat_categoricals.index(name)] = -999
    changed["decoder_target"].fill_(float("nan"))
    torch.testing.assert_close(model(changed).prediction, expected)
    changed["decoder_cont"][..., dataset.reals.index("known")] += 10
    assert not torch.allclose(model(changed).prediction, expected)


def test_exogenous_basis_projection():
    block = NBEATSxBlock("exogenous", 5, 3, 2, 4, 1, 3, 2, 0)
    with torch.no_grad():
        block.theta_b_fc.weight.zero_()
        block.theta_f_fc.weight.zero_()
        block.theta_b_fc.bias.copy_(torch.tensor([2.0, 3.0]))
        block.theta_f_fc.bias.copy_(torch.tensor([4.0, 5.0]))
    encoder = torch.arange(6.0).reshape(1, 3, 2)
    decoder = torch.arange(4.0).reshape(1, 2, 2)
    backcast, forecast = block(torch.zeros(1, 3), torch.zeros(1, 2), encoder, decoder)
    torch.testing.assert_close(backcast, 2 * encoder[..., 0] + 3 * encoder[..., 1])
    torch.testing.assert_close(forecast, 4 * decoder[..., 0] + 5 * decoder[..., 1])


@pytest.mark.parametrize("sharing", [True, False])
def test_weight_sharing(dataset, sharing):
    model = make_model(
        dataset, stack_types=["generic", "generic"], num_blocks=[2], sharing=[sharing]
    )
    assert (model.net_blocks[0] is model.net_blocks[1]) == sharing
    assert model.net_blocks[1] is not model.net_blocks[2]
    assert (model.net_blocks[2] is model.net_blocks[3]) == sharing


@pytest.mark.parametrize(
    "features", ["target_only", "static_only", "historical_only", "grouped"]
)
def test_feature_combinations(features):
    frame = pd.DataFrame(
        {
            "time": np.tile(np.arange(30), 2),
            "group": np.repeat(["a", "b"], 30),
            "target": np.arange(60, dtype=float),
            "known": np.tile(np.arange(30, dtype=float), 2),
            "calendar": np.tile(["a", "b", "c"], 20),
            "event": np.tile(["a", "b"], 30),
        }
    )
    options = {}
    if features == "static_only":
        options["static_categoricals"] = ["group"]
    elif features == "historical_only":
        options["time_varying_unknown_categoricals"] = ["event"]
    elif features == "grouped":
        options.update(
            variable_groups={"holidays": ["calendar", "event"]},
            time_varying_known_categoricals=["holidays"],
            time_varying_known_reals=["known"],
        )
    alternate = TimeSeriesDataSet(
        frame,
        time_idx="time",
        group_ids=["group"],
        target="target",
        max_encoder_length=6,
        max_prediction_length=3,
        time_varying_unknown_reals=["target"],
        **options,
    )
    model = make_model(alternate, num_blocks=[1])
    x, _ = next(iter(alternate.to_dataloader(train=False, batch_size=2)))
    assert torch.isfinite(model(x).prediction).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"stack_types": []}, "stack_types"),
        ({"stack_types": ["invalid"]}, "stack_types"),
        ({"num_blocks": [0]}, "positive integers"),
        ({"num_blocks": [1, 2]}, "length 1"),
        ({"context_length": 5}, "match the dataset"),
        ({"loss": QuantileLoss()}, "point losses"),
    ],
)
def test_invalid_configuration(dataset, kwargs, match):
    with pytest.raises(ValueError, match=match):
        NBeatsX.from_dataset(dataset, **kwargs)


def test_invalid_dataset(dataset):
    variable_length = copy.copy(dataset)
    variable_length.min_encoder_length = 2
    with pytest.raises(ValueError, match="fixed encoder"):
        make_model(variable_length)
    multiple_targets = copy.copy(dataset)
    multiple_targets.target = ["target"]
    with pytest.raises(ValueError, match="single continuous target"):
        make_model(multiple_targets)
    with pytest.raises(ValueError, match="both encoder and decoder"):
        make_model(
            dataset,
            stack_types=["exogenous"],
            allowed_encoder_known_variable_names=[],
        )
    x, _ = next(iter(dataset.to_dataloader(train=False, batch_size=2)))
    model = make_model(dataset, num_blocks=[1])
    x["encoder_lengths"][0] -= 1
    with pytest.raises(ValueError, match="fixed encoder"):
        model(x)


@pytest.mark.parametrize(
    "stack_types", [["generic"], ["trend", "seasonality", "exogenous"]]
)
def test_training_checkpoint_and_prediction(dataset, tmp_path, stack_types):
    loader = dataset.to_dataloader(train=False, batch_size=4)
    model = make_model(dataset, stack_types=stack_types, num_blocks=[2], sharing=[True])
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    checkpoint = tmp_path / "nbeatsx.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = NBeatsX.load_from_checkpoint(checkpoint).eval()
    model.eval()
    x, _ = next(iter(loader))
    torch.testing.assert_close(restored(x).prediction, model(x).prediction)
    assert restored.net_blocks[0] is restored.net_blocks[1]
    prediction = restored.predict(
        loader,
        fast_dev_run=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    assert prediction.shape == (4, 3)
    assert torch.isfinite(prediction).all()


def test_package_registration():
    assert NBeatsX._pkg() is NBeatsX_pkg
    assert NBeatsX_pkg.get_cls() is NBeatsX
    assert NBeatsX_pkg.get_class_tag("capability:exogenous")
