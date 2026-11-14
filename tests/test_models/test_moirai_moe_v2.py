"""Moirai-MoE adapter tests; no pretrained downloads are required."""

import builtins
import importlib.util
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.models import MoiraiMoE, MoiraiMoE_pkg_v2
from pytorch_forecasting.models.moirai import _moirai_moe_v2 as adapter


@pytest.fixture
def metadata():
    return {
        "context_length": 8,
        "prediction_length": 3,
        "n_features": {"target": 1},
    }


@pytest.fixture
def backend(monkeypatch):
    """Exercise the adapter without requiring the optional dependency."""

    class Backbone(nn.Module):
        patch_sizes = (16,)

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    class Forecast(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.module = kwargs.pop("module")
            self.config = kwargs

        def forward(self, **kwargs):
            self.inputs = kwargs
            self.was_training = self.training
            target = kwargs["past_target"]
            batch, _, dim = target.shape
            samples = torch.arange(
                self.config["num_samples"], device=target.device, dtype=target.dtype
            ).view(1, -1, 1, 1)
            samples = samples.expand(batch, -1, self.config["prediction_length"], dim)
            return samples.squeeze(-1) if dim == 1 else samples

    Backbone.from_pretrained = Mock(side_effect=lambda _: Backbone())
    module = ModuleType("uni2ts.model.moirai_moe")
    module.MoiraiMoEModule = Backbone
    module.MoiraiMoEForecast = Forecast
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai_moe", module)
    monkeypatch.setattr(adapter, "_check_soft_dependencies", Mock())
    return Backbone


def test_pretrained_loading(backend, metadata):
    model = MoiraiMoE(model_id="local-model", metadata=metadata, num_samples=5)
    backend.from_pretrained.assert_called_once_with("local-model")
    assert model.forecast.config == {
        "context_length": 8,
        "prediction_length": 3,
        "target_dim": 1,
        "feat_dynamic_real_dim": 0,
        "past_feat_dynamic_real_dim": 0,
        "patch_size": 16,
        "num_samples": 5,
    }
    assert not any(p.requires_grad for p in model.forecast.parameters())


def test_local_module(backend, metadata):
    module = backend()
    model = MoiraiMoE(module=module, metadata=metadata)
    assert model.forecast.module is module
    backend.from_pretrained.assert_not_called()


@pytest.mark.parametrize("target_dim", [1, 2])
def test_samples_and_quantiles(backend, metadata, target_dim):
    metadata["n_features"]["target"] = target_dim
    model = MoiraiMoE(metadata=metadata, num_samples=5)
    out = model({"history_target": torch.ones(2, 8, target_dim)})
    assert out["samples"].shape == (2, 5, 3, target_dim)
    assert out["prediction"].shape == (2, 3, target_dim)
    torch.testing.assert_close(out["prediction"], torch.full((2, 3, target_dim), 2.0))
    point_shape = (2, 3) if target_dim == 1 else (2, 3, target_dim)
    assert model.to_prediction(out).shape == point_shape
    quantiles = model.to_quantiles(out, [0.0, 0.5, 1.0])
    assert quantiles.shape == (*point_shape, 3)
    torch.testing.assert_close(
        quantiles, torch.tensor([0.0, 2.0, 4.0]).expand_as(quantiles)
    )
    assert model.to_quantiles(out).shape == (*point_shape, 7)


def test_missing_and_masked_targets(backend, metadata):
    model = MoiraiMoE(metadata=metadata)
    target = torch.ones(2, 8)
    target[0, 1] = float("nan")
    target[1, 2] = float("inf")
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 0] = False
    original = target.clone()
    model.train()
    model({"history_target": target, "history_mask": valid})
    inputs = model.forecast.inputs
    expected = torch.isfinite(target).unsqueeze(-1) & valid.unsqueeze(-1)
    assert torch.equal(inputs["past_observed_target"], expected)
    assert torch.equal(inputs["past_is_pad"], ~valid)
    assert torch.isfinite(inputs["past_target"]).all()
    assert (inputs["past_target"][~expected] == 0).all()
    assert not model.forecast.was_training
    torch.testing.assert_close(target, original, equal_nan=True)


@pytest.mark.parametrize("batched_scale", [False, True])
def test_restore_scale(backend, metadata, batched_scale):
    metadata["n_features"]["target"] = 2
    model = MoiraiMoE(metadata=metadata, num_samples=5)
    scale = torch.tensor([2.0, 3.0])
    center = torch.tensor([10.0, 20.0])
    if batched_scale:
        scale = torch.stack([scale, scale * 2])
        center = torch.stack([center, center * 2])
    out = model(
        {
            "history_target": torch.ones(2, 8, 2),
            "target_scale": {"scale": scale, "center": center},
        }
    )
    expected = (2 * scale + center).expand(2, 2).unsqueeze(1).expand(2, 3, 2)
    torch.testing.assert_close(out["prediction"], expected)
    torch.testing.assert_close(model.to_quantiles(out, [0.5]).squeeze(-1), expected)


@pytest.mark.parametrize("name", ["context_length", "prediction_length", "target_dim"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_metadata(metadata, name, value):
    if name == "target_dim":
        metadata["n_features"]["target"] = value
    else:
        metadata[name] = value
    with pytest.raises(ValueError, match=name):
        MoiraiMoE(metadata=metadata)


@pytest.mark.parametrize("kwargs", [{"num_samples": 0}, {"patch_size": 0}])
def test_invalid_options(metadata, kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        MoiraiMoE(metadata=metadata, **kwargs)


def test_unsupported_patch_size(backend, metadata):
    with pytest.raises(ValueError, match="patch_size must be one of"):
        MoiraiMoE(metadata=metadata, patch_size=8)


@pytest.mark.parametrize("shape", [(2, 7, 1), (2, 8, 2), (8,), (2, 8, 1, 1)])
def test_invalid_history(backend, metadata, shape):
    model = MoiraiMoE(metadata=metadata)
    with pytest.raises(ValueError, match="history_target"):
        model({"history_target": torch.ones(shape)})


@pytest.mark.parametrize("mask", [torch.ones(2, 8), torch.ones(2, 7, dtype=torch.bool)])
def test_invalid_mask(backend, metadata, mask):
    model = MoiraiMoE(metadata=metadata)
    with pytest.raises(ValueError, match="history_mask"):
        model({"history_target": torch.ones(2, 8, 1), "history_mask": mask})


@pytest.mark.parametrize("quantiles", [[], [-0.1], [1.1], [float("nan")], [[0.5]]])
def test_invalid_quantiles(backend, metadata, quantiles):
    model = MoiraiMoE(metadata=metadata)
    out = model({"history_target": torch.ones(2, 8, 1)})
    with pytest.raises(ValueError, match="quantiles"):
        model.to_quantiles(out, quantiles)


def test_missing_dependency(monkeypatch, metadata):
    check = Mock(side_effect=ModuleNotFoundError("uni2ts is required"))
    monkeypatch.setattr(adapter, "_check_soft_dependencies", check)
    with pytest.raises(ModuleNotFoundError, match="uni2ts"):
        MoiraiMoE(metadata=metadata)
    check.assert_called_once_with("uni2ts>=2.0,<3", obj=check.call_args.kwargs["obj"])


def test_import_without_uni2ts(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("uni2ts"):
            raise AssertionError("uni2ts must not be imported at module scope")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location(
        "moirai_optional_import", adapter.__file__
    )
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_training_rejected(backend, metadata):
    model = MoiraiMoE(metadata=metadata)
    assert model.configure_optimizers() is None
    with pytest.raises(NotImplementedError, match="zero-shot"):
        model.training_step(None, 0)
    with pytest.raises(NotImplementedError, match="zero-shot"):
        MoiraiMoE_pkg_v2().fit(None)


def test_package_linkage():
    assert MoiraiMoE.pkg is MoiraiMoE_pkg_v2
    assert MoiraiMoE_pkg_v2.get_cls() is MoiraiMoE
    assert MoiraiMoE_pkg_v2.get_datamodule_cls() is TslibDataModule
    assert not MoiraiMoE_pkg_v2.get_class_tag("capability:exogenous")


@pytest.fixture
def dataset():
    return TimeSeries(
        pd.DataFrame({"time": np.arange(60), "value": np.arange(60, dtype=float)}),
        time="time",
        target=["value"],
    )


@pytest.mark.parametrize("use_datamodule", [False, True])
def test_package_prediction(backend, dataset, use_datamodule):
    config = {"context_length": 8, "prediction_length": 3, "batch_size": 4}
    pkg = MoiraiMoE_pkg_v2(
        model_cfg={"num_samples": 5},
        datamodule_cfg=config,
        trainer_cfg={
            "accelerator": "cpu",
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    data = TslibDataModule(dataset, **config) if use_datamodule else dataset
    for mode, shape in [("prediction", (3,)), ("quantiles", (3, 7)), ("raw", (3, 1))]:
        out = pkg.predict(data, mode=mode)
        assert out["prediction"].shape[1:] == shape
        assert torch.isfinite(out["prediction"]).all()
        if mode == "raw":
            assert out["samples"].shape[1:] == (5, 3, 1)
    backend.from_pretrained.assert_called_once()
    out = pkg.predict(pkg.datamodule.predict_dataloader())
    assert out["prediction"].shape[-1] == 3


def test_first_prediction_needs_metadata():
    with pytest.raises(ValueError, match="first prediction requires"):
        MoiraiMoE_pkg_v2().predict([])


@pytest.mark.parametrize("field", ["context_length", "prediction_length", "target"])
def test_changed_metadata_rejected(backend, metadata, dataset, field):
    pkg = MoiraiMoE_pkg_v2()
    pkg.model = MoiraiMoE(metadata=metadata)
    data = TslibDataModule(dataset, context_length=8, prediction_length=3)
    if field == "target":
        data.metadata["n_features"][field] += 1
    else:
        data.metadata[field] += 1
    with pytest.raises(ValueError, match="must match"):
        pkg.predict(data)


@pytest.mark.parametrize("target_dim,prediction_length", [(1, 4), (2, 20)])
def test_real_uni2ts_local_weights(tmp_path, target_dim, prediction_length):
    """Test real sampling, including autoregressive horizons, without network access."""
    pytest.importorskip("uni2ts.model.moirai_moe")
    from uni2ts.distribution import StudentTOutput
    from uni2ts.model.moirai_moe import MoiraiMoEModule

    module = MoiraiMoEModule(
        distr_output=StudentTOutput(),
        d_model=64,
        d_ff=64,
        num_layers=1,
        patch_sizes=(16,),
        max_seq_len=32,
        attn_dropout_p=0.0,
        dropout_p=0.0,
    )
    module.save_pretrained(tmp_path)
    model = MoiraiMoE(
        model_id=str(tmp_path),
        num_samples=3,
        metadata={
            "context_length": 32,
            "prediction_length": prediction_length,
            "n_features": {"target": target_dim},
        },
    )
    out = model({"history_target": torch.randn(2, 32, target_dim)})
    assert out["samples"].shape == (2, 3, prediction_length, target_dim)
    assert torch.isfinite(out["samples"]).all()
    assert not out["samples"].requires_grad
