import os
import sys
import typing as ty

import optuna
import pytest

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import LSTMModel
from pytorch_forecasting.models.tuning import optimize_hyperparameters


def test_tuning_lst(timeseriesdataset_multitarget: TimeSeriesDataSet) -> optuna.Study:
    """Test we can tune a `LSTMModel` model."""
    # create dataloaders for model
    batch_size = 32
    train_dataloader = timeseriesdataset_multitarget.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = timeseriesdataset_multitarget.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    # Create HP to explore
    hp: ty.Dict[str, ty.Dict[str, ty.Any]] = {
        "n_layers": {
            "method": "suggest_int",
            "ranges": (1, 8),
        },
        "hidden_size": {
            "method": "suggest_int",
            "ranges": (4, 16),
        },
        "dropout": {
            "method": "suggest_uniform",
            "ranges": (0.1, 0.3),
        },
    }
    # create study
    model_class = LSTMModel
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        monitor="val_loss",
        model_path=os.path.join("pytest_artifacts", f"hpo_{model_class.__name__}"),
        model_class=model_class,
        input_params=hp,
        n_trials=2,
        max_epochs=10,
        gradient_clip_val_range=(0.01, 1.0),
        learning_rate_range=(0.001, 0.1),
        trainer_kwargs=dict(limit_train_batches=30, accelerator="cpu"),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=True,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )
    return study


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
