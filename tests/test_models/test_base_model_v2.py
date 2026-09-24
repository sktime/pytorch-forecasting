import warnings

import pytest
import torch

from pytorch_forecasting.metrics import MAE, MultiLoss
from pytorch_forecasting.metrics.nn_loss_adapter import NNLossAdapter
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class DummyModel(BaseModel):
    """Minimal concrete subclass for testing optimizer/scheduler wiring."""

    def __init__(self, **kwargs):
        kwargs.setdefault("loss", MAE())
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return {"prediction": self.linear(x["x"])}


def _make_model(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DummyModel(**kwargs)


# --- optimizer tests ---


def test_optimizer_generic_torch_optim_lookup():
    """Optimizer names not in the registry fall back to torch.optim by class name."""
    model = _make_model(optimizer="RMSprop", optimizer_params={"lr": 1e-3})
    cfg = model.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.RMSprop)


def test_optimizer_callable():
    """Passing an optimizer class directly bypasses the registry lookup."""
    model = _make_model(optimizer=torch.optim.AdamW, optimizer_params={"lr": 1e-3})
    cfg = model.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)


@pytest.mark.parametrize(
    "bad_optimizer,match",
    [
        ("not_a_real_optimizer", "not supported"),
        (12345, "must be a string"),
    ],
)
def test_optimizer_invalid_input(bad_optimizer, match):
    """Invalid optimizer values raise ValueError with a descriptive message."""
    model = _make_model(optimizer=bad_optimizer)
    with pytest.raises(ValueError, match=match):
        model.configure_optimizers()


# --- scheduler tests ---


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("reduce_lr_on_plateau", torch.optim.lr_scheduler.ReduceLROnPlateau),
        ("step_lr", torch.optim.lr_scheduler.StepLR),
        ("cosine_annealing", torch.optim.lr_scheduler.CosineAnnealingLR),
        (
            "cosine_annealing_warm_restarts",
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        ),
    ],
)
def test_scheduler_registry_strings(name, expected_cls):
    """Each scheduler registry key resolves to the expected scheduler class."""
    sched_params = {"step_size": 10} if name == "step_lr" else {}
    if name == "cosine_annealing":
        sched_params["T_max"] = 50
    elif name == "cosine_annealing_warm_restarts":
        sched_params["T_0"] = 10
    model = _make_model(
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler=name,
        lr_scheduler_params=sched_params,
    )
    cfg = model.configure_optimizers()
    if isinstance(cfg.get("lr_scheduler"), dict):
        sched = cfg["lr_scheduler"]["scheduler"]
    else:
        sched = cfg["lr_scheduler"]
    assert isinstance(sched, expected_cls)


def test_scheduler_invalid_string():
    """An unrecognised scheduler string raises ValueError."""
    model = _make_model(
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="bogus_scheduler",
    )
    with pytest.raises(ValueError, match="not supported"):
        model.configure_optimizers()


def test_reduce_lr_on_plateau_returns_monitor():
    """ReduceLROnPlateau config includes the monitor key required by Lightning."""
    model = _make_model(
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
    )
    cfg = model.configure_optimizers()
    assert "lr_scheduler" in cfg
    assert cfg["lr_scheduler"]["monitor"] == "val_loss"


def test_optimizer_instance():
    """A pre-built optimizer instance is passed through without modification."""
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    model.optimizer = opt
    cfg = model.configure_optimizers()
    assert cfg["optimizer"] is opt


# --- MultiLoss / multi-target tests ---


class MultiTargetDummyModel(BaseModel):
    """Dummy model returning one prediction tensor per target.

    Mirrors what a multi-target v2 model emits: ``prediction`` is a ``list``
    with one ``[batch, horizon]`` tensor per target.
    """

    def __init__(self, n_targets: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        self.linear = torch.nn.Linear(1, n_targets)

    def forward(self, x):
        # [batch, horizon, n_targets] -> list of n_targets x [batch, horizon]
        out = self.linear(x["x"])
        return {"prediction": [out[..., i] for i in range(self.n_targets)]}


class StackedDummyModel(BaseModel):
    """Dummy model returning a single stacked ``[batch, horizon, n_targets]``."""

    def __init__(self, n_targets: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        self.linear = torch.nn.Linear(1, n_targets)

    def forward(self, x):
        return {"prediction": self.linear(x["x"])}


def _make(cls, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cls(**kwargs)


def _batch(n_targets: int, batch_size: int = 4, horizon: int = 3):
    """Synthetic ``(x, y)`` batch in the v2 datamodule format.

    ``y`` is a bare tensor for a single target and a ``list`` of tensors for
    multiple targets - no weights, matching what the v2 collate functions emit.
    """
    x = {"x": torch.randn(batch_size, horizon, 1)}
    targets = [torch.randn(batch_size, horizon) for _ in range(n_targets)]
    y = targets if n_targets > 1 else targets[0]
    return x, y


def test_single_target_plain_loss():
    """A single target with a plain metric trains through the shared _step."""
    model = _make(DummyModel, loss=MAE())
    x = {"x": torch.randn(4, 3, 1)}
    y = torch.randn(4, 3)
    # DummyModel emits [batch, horizon, 1], which MAE squeezes internally
    out = model._step((x, y), 0)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_multi_target_with_multiloss():
    """Two targets with a MultiLoss produce a finite scalar loss."""
    model = _make(MultiTargetDummyModel, n_targets=2, loss=MultiLoss([MAE(), MAE()]))
    out = model._step(_batch(2), 0)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_multiloss_sums_per_target_losses():
    """The MultiLoss result equals the sum of the per-target metric values.

    This pins the y_actual[0][idx] / y_actual[1] indexing contract: if the
    bare target list were passed through unwrapped, MultiLoss would index a
    batch row instead of a target and this equality would not hold.
    """
    model = _make(MultiTargetDummyModel, n_targets=2, loss=MultiLoss([MAE(), MAE()]))
    x, y = _batch(2)
    out = model._step((x, y), 0)

    preds = model(x)["prediction"]
    expected = MAE()(preds[0], y[0]) + MAE()(preds[1], y[1])
    assert torch.allclose(out["loss"], expected)


def test_stacked_prediction_split_for_multiloss():
    """A stacked [batch, horizon, n_targets] output is split per target."""
    model = _make(StackedDummyModel, n_targets=2, loss=MultiLoss([MAE(), MAE()]))
    out = model._step(_batch(2), 0)
    assert torch.isfinite(out["loss"])


def test_target_count_mismatch_raises():
    """A MultiLoss with the wrong number of metrics fails with a clear error."""
    model = _make(
        MultiTargetDummyModel, n_targets=3, loss=MultiLoss([MAE(), MAE(), MAE()])
    )
    with pytest.raises(ValueError, match="MultiLoss holds 3 metrics"):
        model._coerce_y_for_loss(_batch(2)[1])


def test_multiple_targets_with_plain_loss_raises():
    """Multiple targets paired with a non-MultiLoss loss fail early."""
    model = _make(MultiTargetDummyModel, n_targets=2, loss=MAE())
    with pytest.raises(ValueError, match="requires the loss to be a MultiLoss"):
        model._step(_batch(2), 0)


def test_prediction_count_mismatch_raises():
    """A model returning more predictions than the loss has metrics raises."""
    model = _make(MultiTargetDummyModel, n_targets=3, loss=MultiLoss([MAE(), MAE()]))
    with pytest.raises(ValueError, match="MultiLoss holds 2 metrics"):
        model._coerce_y_hat_for_loss([torch.randn(4, 3) for _ in range(3)])


def test_nn_loss_inside_multiloss_is_adapted():
    """A bare nn.MSELoss inside a MultiLoss is wrapped by NNLossAdapter."""
    model = _make(
        MultiTargetDummyModel,
        n_targets=2,
        loss=MultiLoss([torch.nn.MSELoss(), MAE()]),
    )
    assert isinstance(model._loss[0], NNLossAdapter)
    out = model._step(_batch(2), 0)
    assert torch.isfinite(out["loss"])


def test_to_prediction_returns_list_for_multiloss():
    """to_prediction yields one point forecast per target under a MultiLoss."""
    model = _make(MultiTargetDummyModel, n_targets=2, loss=MultiLoss([MAE(), MAE()]))
    x, _ = _batch(2)
    preds = model.to_prediction(model(x))
    assert isinstance(preds, list)
    assert len(preds) == 2


def test_to_quantiles_returns_list_for_multiloss():
    """to_quantiles yields one quantile forecast per target under a MultiLoss."""
    model = _make(MultiTargetDummyModel, n_targets=2, loss=MultiLoss([MAE(), MAE()]))
    x, _ = _batch(2)
    quantiles = model.to_quantiles(model(x))
    assert isinstance(quantiles, list)
    assert len(quantiles) == 2


def test_single_target_to_prediction_unchanged():
    """Single-target to_prediction still returns a bare tensor."""
    model = _make(DummyModel, loss=MAE())
    x = {"x": torch.randn(4, 3, 1)}
    pred = model.to_prediction(model(x))
    assert isinstance(pred, torch.Tensor)


def test_log_metrics_per_target(monkeypatch):
    """Logging metrics are computed once per target with a target-prefixed tag."""
    model = _make(
        MultiTargetDummyModel,
        n_targets=2,
        loss=MultiLoss([MAE(), MAE()]),
        logging_metrics=[MAE()],
    )
    logged = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, **kw: logged.update({name: value})
    )

    out = model._step(_batch(2), 0)
    model.log_metrics(out["y_hat"], out["y"], prefix="val")

    assert "target0_val_MAE" in logged
    assert "target1_val_MAE" in logged


def test_2d_output_not_split_across_targets():
    """A [batch, horizon] output is never split along the horizon dimension.

    With two metrics and a horizon that happens to equal two, splitting the
    last dimension would slice time steps instead of targets - that has to
    fail loudly rather than silently train on the wrong tensors.
    """
    model = _make(StackedDummyModel, n_targets=2, loss=MultiLoss([MAE(), MAE()]))
    with pytest.raises(ValueError, match=r"\(batch, horizon, 2\)"):
        model._coerce_y_hat_for_loss(torch.randn(4, 2))
