import warnings

import pytest
import torch

from pytorch_forecasting.metrics import MAE, NormalDistributionLoss, QuantileLoss
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


def test_step_output_size_point_loss():
    """Point loss (MAE) produces a single output per timestep."""
    model = _make_model(loss=MAE())
    assert model.step_output_size == 1


def test_step_output_size_quantile_loss():
    """QuantileLoss with 3 quantiles needs 3 outputs per timestep."""
    model = _make_model(loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]))
    assert model.step_output_size == 3


def test_step_output_size_distribution_loss():
    """DistributionLoss needs multiple outputs per timestep (e.g. 2 for Normal)."""
    model = _make_model(loss=NormalDistributionLoss())
    assert model.step_output_size == 2


def test_transform_output_none_is_identity():
    """No target_scale means predictions pass through untouched."""
    model = _make_model()
    raw = torch.randn(2, 6, 1)
    assert torch.equal(model.transform_output(raw, None), raw)


def test_transform_output_dict_target_scale():
    """Dict target_scale applies affine denormalization correctly."""
    model = _make_model()
    raw = torch.randn(4, 12, 1)
    center = torch.tensor([10.0, 20.0, 30.0, 40.0])
    scale = torch.tensor([2.0, 3.0, 4.0, 5.0])
    target_scale = {"center": center, "scale": scale}

    result = model.transform_output(raw, target_scale)

    assert result.shape == raw.shape
    assert torch.allclose(result[0], raw[0] * 2.0 + 10.0)
    assert torch.allclose(result[3], raw[3] * 5.0 + 40.0)


def test_transform_output_plain_tensor_target_scale():
    """Plain tensor target_scale correctly applies affine denormalization."""
    model = _make_model()
    raw = torch.randn(4, 12, 1)
    target_scale = torch.tensor([[10.0, 2.0], [20.0, 3.0], [30.0, 4.0], [40.0, 5.0]])

    result = model.transform_output(raw, target_scale)

    assert result.shape == raw.shape
    assert torch.allclose(result[0], raw[0] * 2.0 + 10.0)
    assert torch.allclose(result[2], raw[2] * 4.0 + 30.0)
