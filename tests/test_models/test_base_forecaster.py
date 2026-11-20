"""Tests for the ``fit`` behaviour of :class:`BaseForecaster`."""

from lightning import Trainer
import pytest

from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.dlinear._dlinear_forecaster_v2 import DLinearForecaster

DATAMODULE_CFG = {
    "train_val_test_split": (0.5, 0.5),
    "batch_size": 2,
    "context_length": 8,
    "prediction_length": 2,
}
ARTIFACTS = ["datamodule_cfg.pkl", "model_cfg.pkl", "metadata.pkl"]


def _make_forecaster(tmp_path, **trainer_kwargs):
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        **trainer_kwargs,
    )
    return DLinearForecaster(
        loss=SMAPE(),
        trainer=trainer,
        datamodule=DLinearForecaster.get_datamodule_cls()(**DATAMODULE_CFG),
    )


@pytest.fixture
def train_data():
    return DLinearForecaster.get_test_dataset_from(**DATAMODULE_CFG)["train"]


def test_fit_saves_artifacts_next_to_checkpoint(tmp_path, train_data):
    """With a written checkpoint, artifacts are saved beside it."""
    ckpt_dir = tmp_path / "ckpts"
    forecaster = _make_forecaster(tmp_path, max_epochs=1, limit_train_batches=2)

    best_path = forecaster.fit(train_data, ckpt_dir=ckpt_dir)

    assert best_path is not None and best_path.parent == ckpt_dir
    assert best_path.exists()
    for name in ARTIFACTS:
        assert (ckpt_dir / name).exists()


@pytest.mark.parametrize(
    "trainer_kwargs",
    [{"fast_dev_run": True}, {"max_epochs": 1, "limit_val_batches": 0}],
    ids=["fast_dev_run", "no_validation"],
)
def test_fit_without_checkpoint_does_not_write_to_cwd(
    tmp_path, monkeypatch, train_data, trainer_kwargs
):
    """If no checkpoint is written, ``fit`` must not dump artifacts into the cwd."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    forecaster = _make_forecaster(tmp_path, **trainer_kwargs)

    with pytest.warns(UserWarning, match="no checkpoint was written"):
        result = forecaster.fit(train_data, ckpt_dir=tmp_path / "ckpts")

    assert result is None
    for name in ARTIFACTS:
        assert not (cwd / name).exists()
