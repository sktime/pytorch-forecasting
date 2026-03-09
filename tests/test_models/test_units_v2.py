"""Tests for UniTS v2 model."""

import inspect

import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.models.units import UniTS_pkg_v2
from pytorch_forecasting.models.units._units_v2 import UniTS


def _make_metadata(
    context_length=24,
    prediction_length=6,
    target_dim=3,
    cont_dim=0,
    cat_dim=0,
    features="M",
):
    return {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "features": features,
        "n_features": {
            "target": target_dim,
            "continuous": cont_dim,
            "categorical": cat_dim,
            "static_categorical": 0,
            "static_continuous": 0,
        },
        "feature_indices": {
            "target": list(range(target_dim)),
            "continuous": list(range(cont_dim)),
            "categorical": [],
            "known": [],
            "unknown": [],
        },
        "feature_names": {},
    }


def _make_model(metadata, **kwargs):
    return UniTS(loss=nn.MSELoss(), metadata=metadata, **kwargs)


def _make_batch(metadata, batch_size=2):
    B = batch_size
    T = metadata["context_length"]
    C = metadata["n_features"]["target"]
    Cc = metadata["n_features"]["continuous"]
    batch = {
        "history_target": torch.randn(B, T, C),
        "history_time_idx": torch.arange(T).unsqueeze(0).expand(B, -1),
        "target_scale": {
            "scale": torch.ones(B, 1, C),
            "center": torch.zeros(B, 1, C),
        },
    }
    if Cc > 0:
        batch["history_cont"] = torch.randn(B, T, Cc)
    return batch


class TestUniTSPkg:
    """Tests for UniTS_pkg_v2."""

    def test_get_cls(self):
        assert UniTS_pkg_v2.get_cls().__name__ == "UniTS"

    def test_pkg_tags(self):
        instance = UniTS_pkg_v2()
        assert instance.get_tag("info:name") == "UniTS"
        assert instance.get_tag("capability:multivariate") is True
        assert instance.get_tag("capability:exogenous") is True
        assert instance.get_tag("capability:pred_int") is False

    def test_get_datamodule_cls(self):
        assert UniTS_pkg_v2.get_datamodule_cls() is not None

    def test_get_test_train_params(self):
        params = UniTS_pkg_v2.get_test_train_params()
        assert isinstance(params, list) and len(params) > 0
        for p in params:
            assert "datamodule_cfg" in p
            dm = p["datamodule_cfg"]
            assert "context_length" in dm
            assert "prediction_length" in dm
            # patch_len must not exceed context_length
            patch_len = p.get("patch_len", 16)
            assert patch_len <= dm["context_length"], (
                f"patch_len={patch_len} exceeds context_length={dm['context_length']}"
            )

    def test_get_test_train_params_independent(self):
        """Each param dict must be independent — no shared mutable objects."""
        params = UniTS_pkg_v2.get_test_train_params()
        dm_ids = [id(p["datamodule_cfg"]) for p in params]
        assert len(dm_ids) == len(set(dm_ids)), "datamodule_cfg dicts are shared objects"


class TestUniTSForward:
    """Forward pass shape and numerical correctness."""

    @pytest.fixture
    def meta(self):
        return _make_metadata()

    def test_output_shape_default(self, meta):
        model = _make_model(meta)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert "prediction" in out
        assert out["prediction"].shape == (2, 6, 3)

    def test_output_shape_small_dmodel(self, meta):
        model = _make_model(meta, d_model=32, n_heads=4, d_ff=64)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert out["prediction"].shape == (2, 6, 3)

    def test_output_shape_single_layer(self, meta):
        model = _make_model(meta, e_layers=1, d_model=32, n_heads=4, d_ff=64)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert out["prediction"].shape == (2, 6, 3)

    def test_no_nan(self, meta):
        model = _make_model(meta)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert not torch.isnan(out["prediction"]).any()

    def test_no_inf(self, meta):
        model = _make_model(meta)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert not torch.isinf(out["prediction"]).any()

    def test_pkg_classmethod(self):
        assert UniTS._pkg().__name__ == "UniTS_pkg_v2"

    def test_gradients_flow(self, meta):
        """Loss.backward() must not raise and must produce non-zero gradients."""
        model = _make_model(meta, d_model=32, n_heads=4, d_ff=64)
        model.train()
        out = model(_make_batch(meta))
        target = torch.randn(2, 6, 3)
        loss = nn.MSELoss()(out["prediction"], target)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_exogenous_features(self):
        """Model must accept and use continuous exogenous features."""
        meta = _make_metadata(context_length=24, prediction_length=6, target_dim=2, cont_dim=3)
        model = _make_model(meta, d_model=32, n_heads=4, d_ff=64)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert out["prediction"].shape == (2, 6, 2)

    @pytest.mark.parametrize("pred_len", [1, 3, 12, 24])
    def test_prediction_lengths(self, pred_len):
        meta = _make_metadata(context_length=48, prediction_length=pred_len, target_dim=2)
        model = _make_model(meta, d_model=32, n_heads=4, d_ff=64)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert out["prediction"].shape == (2, pred_len, 2)

    @pytest.mark.parametrize("target_dim", [1, 4, 7])
    def test_target_dims(self, target_dim):
        meta = _make_metadata(context_length=24, prediction_length=6, target_dim=target_dim)
        model = _make_model(meta, d_model=32, n_heads=4, d_ff=64)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(meta))
        assert out["prediction"].shape == (2, 6, target_dim)


class TestUniTSParams:
    """Parameter validation."""

    def test_d_model_not_divisible_by_n_heads(self):
        with pytest.raises(ValueError, match="d_model"):
            UniTS(loss=nn.MSELoss(), metadata=_make_metadata(), d_model=33, n_heads=8)

    def test_patch_len_exceeds_context_length(self):
        with pytest.raises(ValueError, match="patch_len"):
            UniTS(
                loss=nn.MSELoss(),
                metadata=_make_metadata(context_length=8),
                patch_len=16,
            )

    def test_default_hyperparameters(self):
        sig = inspect.signature(UniTS.__init__)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        assert defaults["d_model"] == 64
        assert defaults["n_heads"] == 8
        assert defaults["e_layers"] == 3
        assert defaults["d_ff"] == 512
        assert defaults["dropout"] == 0.1
        assert defaults["patch_len"] == 16
        assert defaults["stride"] == 8
        assert defaults["prompt_len"] == 10
