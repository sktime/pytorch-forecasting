import warnings

import pytest
import torch

from pytorch_forecasting.metrics import MAE
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


class BoundaryModel(BaseModel):
    """Expose supplied predictions to test metric boundaries directly."""

    def forward(self, x):
        return {"prediction": x["prediction"]}


def _boundary_model(loss, logging_metrics=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return BoundaryModel(loss=loss, logging_metrics=logging_metrics)


@pytest.mark.parametrize("stage", ["training_step", "validation_step", "test_step"])
@pytest.mark.parametrize("n_targets", [1, 2])
def test_tensor_loss_boundary(stage, n_targets):
    """Split targets rather than quantiles, retaining loss values and gradients."""
    from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

    loss = MultiLoss([QuantileLoss([0.1, 0.9]) for _ in range(n_targets)])
    model = _boundary_model(loss)
    prediction = (
        torch.arange(1, 12 * n_targets + 1, dtype=torch.float32)
        .reshape(2, 3, n_targets, 2)
        .requires_grad_()
    )
    target = torch.zeros(2, 3, n_targets)
    result = getattr(model, stage)(({"prediction": prediction}, target), 0)
    key = {
        "training_step": "loss",
        "validation_step": "val_loss",
        "test_step": "test_loss",
    }[stage]
    # All predictions are positive: pinball loss is 2 * (1-q) * prediction.
    expected = (
        (2 * prediction * prediction.new_tensor([0.9, 0.1])).mean((0, 1, 3)).sum()
    )
    torch.testing.assert_close(result[key], expected)
    torch.testing.assert_close(
        torch.autograd.grad(result[key], prediction, retain_graph=True)[0],
        torch.autograd.grad(expected, prediction)[0],
    )


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("quantiles", [[0.5], [0.1, 0.5, 0.9]])
def test_prediction_boundary_preserves_metric_conversion(multi, quantiles):
    """Restore K=1 and K=Q without replacing bound conversion semantics."""
    from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

    n_targets = len(quantiles) if multi else 1
    loss = (
        MultiLoss([QuantileLoss(quantiles) for _ in range(n_targets)])
        if multi
        else QuantileLoss(quantiles)
    )
    model = _boundary_model(loss)
    values = [7.0] if len(quantiles) == 1 else [-10.0, 2.0, 50.0]
    prediction = torch.tensor(values).expand(2, 4, n_targets, -1)
    torch.testing.assert_close(
        model.to_quantiles({"prediction": prediction}, unused=True), prediction
    )
    torch.testing.assert_close(
        model.to_prediction({"prediction": prediction}, unused=True),
        prediction[..., quantiles.index(0.5)],
    )
    target = torch.zeros(2, 4, n_targets)
    actual = model.training_step(({"prediction": prediction}, target), 0)["loss"]
    expected = sum(
        QuantileLoss(quantiles)(prediction[:, :, k], target[:, :, k])
        for k in range(n_targets)
    )
    torch.testing.assert_close(actual, expected)


def test_legacy_list_boundary_and_count_check():
    """Wrap legacy targets for MultiLoss and reject missing target predictions."""
    from pytorch_forecasting.metrics import MultiLoss

    model = _boundary_model(MultiLoss([MAE(), MAE()]))
    prediction = [torch.ones(2, 3), torch.full((2, 3), 4.0)]
    targets = [torch.zeros(2, 3), torch.ones(2, 3)]
    actual = model.training_step(({"prediction": prediction}, targets), 0)["loss"]
    torch.testing.assert_close(actual, torch.tensor(4.0))
    result = model.to_prediction({"prediction": prediction})
    assert isinstance(result, list)
    for actual, expected in zip(result, prediction):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="target|prediction|MultiLoss"):
        model.training_step(({"prediction": prediction[:1]}, targets), 0)


def test_logging_preserves_existing_raw_input_semantics(monkeypatch):
    """Adding boundary hooks must not apply the training loss's point transform."""
    from pytorch_forecasting.metrics import PoissonLoss

    model = _boundary_model(PoissonLoss(), logging_metrics=[MAE()])
    logged = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, **kwargs: logged.update({name: value})
    )
    model.log_metrics(torch.zeros(2, 3, 1), torch.ones(2, 3), prefix="val")
    # Legacy logging scores the supplied raw zeros, not exp(zeros).
    torch.testing.assert_close(logged["val_MAE"], torch.tensor(1.0))


def test_logging_uses_its_own_metric_boundary(monkeypatch):
    """A logging MultiLoss receives converted inputs without new per-target tags."""
    from pytorch_forecasting.metrics import MultiLoss

    model = _boundary_model(
        MultiLoss([MAE(), MAE()]), logging_metrics=[MultiLoss([MAE(), MAE()])]
    )
    logged = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, **kwargs: logged.update({name: value})
    )
    prediction = torch.tensor([1.0, 4.0]).expand(2, 3, 2).unsqueeze(-1)
    model.log_metrics(prediction, torch.zeros(2, 3, 2), prefix="val")
    assert set(logged) == {"val_MultiLoss"}
    torch.testing.assert_close(logged["val_MultiLoss"], torch.tensor(5.0))
