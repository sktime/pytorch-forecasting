from typing import Any

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


def _setup_pkg_and_data(
    estimator_cls: type[BaseForecaster],
    trainer_kwargs: dict[str, Any],
    tmp_path: str,
) -> tuple[BaseForecaster, dict[str, TimeSeries], dict[str, Any]]:
    """
    Helper to initialize the Package, Datasets, and Configs.

    Returns
    -------
    pkg : BaseForecaster
        The initialized model package.
    test_data : dict
        Dictionary containing 'train' and 'predict' TimeSeries datasets.
    datamodule_cfg : dict
        The final datamodule configuration used.
    """
    params_copy = trainer_kwargs.copy()
    datamodule_cfg = params_copy.pop("datamodule_cfg", {})
    model_cfg = params_copy

    if "loss" not in model_cfg:
        model_cfg["loss"] = SMAPE()

    default_datamodule_cfg = {
        "train_val_test_split": (0.8, 0.2),
        "add_relative_time_idx": True,
        "batch_size": 2,
    }
    default_datamodule_cfg.update(datamodule_cfg)

    logger = TensorBoardLogger(str(tmp_path))
    trainer_cfg = {
        "max_epochs": 2,
        "gradient_clip_val": 0.1,
        "enable_checkpointing": True,
        "default_root_dir": str(tmp_path),
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "accelerator": "cpu",
        "logger": logger,
    }

    test_data = estimator_cls.get_test_dataset_from(**default_datamodule_cfg)

    pkg = estimator_cls(
        **model_cfg,
        trainer=Trainer(**trainer_cfg),
        datamodule=estimator_cls.get_datamodule_cls()(**default_datamodule_cfg),
    )

    return pkg, test_data, default_datamodule_cfg
