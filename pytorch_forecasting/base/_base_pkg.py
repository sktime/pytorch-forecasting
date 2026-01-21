from pathlib import Path
import pickle
from typing import Any, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.datamodule import LightningDataModule
import torch
from torch.utils.data import DataLoader
import yaml

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class Base_pkg(_BasePtForecasterV2):
    """
    Base model package class acting as a high-level wrapper for the Lightning workflow.

    This class simplifies the user experience by managing model, datamodule, and trainer
    configurations, and providing streamlined ``fit`` and ``predict`` methods.

    Parameters
    ----------
    model_cfg : dict, optional
        Model configs for the initialisation of the model. Required if not loading
        from a checkpoint. Defaults to ``{}``.
    trainer_cfg : dict, optional
        Configs to initialise ``lightning.Trainer``. Defaults to {}.
    datamodule_cfg : Union[dict, str, Path], optional
        Configs to initialise a ``LightningDataModule``.

        - If dict, the keys and values are used as configuration parameters.
        - If str or Path, it should be a path to a ``.pkl`` file containing
          the serialized configuration dictionary. Required for reproducibility
          when loading a model for inference. Defaults to {}.

    ckpt_path : Union[str, Path], optional
        Path to the checkpoint from which to load the model. If provided, `model_cfg`
        is ignored. Defaults to None.
    """

    def __init__(
        self,
        model_cfg: dict[str, Any] | str | Path | None = None,
        trainer_cfg: dict[str, Any] | str | Path | None = None,
        datamodule_cfg: dict[str, Any] | str | Path | None = None,
        ckpt_path: str | Path | None = None,
    ):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.model_cfg = self._load_config(
            model_cfg, ckpt_path=self.ckpt_path, auto_file_name="model_cfg.pkl"
        )
        print(self.model_cfg)

        self.datamodule_cfg = self._load_config(
            datamodule_cfg,
            ckpt_path=self.ckpt_path,
            auto_file_name="datamodule_cfg.pkl",
        )
        self.trainer_cfg = self._load_config(trainer_cfg)
        self.metadata = self._load_config(
            None, ckpt_path=self.ckpt_path, auto_file_name="metadata.pkl"
        )

        self.model = None
        self.trainer = None
        self.datamodule = None
        if self.ckpt_path:
            print(self.metadata)
            self._build_model(metadata=self.metadata, **self.model_cfg)
        else:
            self.model = None

    @staticmethod
    def _load_config(
        config: dict | str | Path | None,
        ckpt_path: str | Path | None = None,
        auto_file_name: str | None = None,
    ) -> dict:
        """
        Loads configuration from a dictionary, YAML file, or Pickle file.
        """
        if config is None:
            if ckpt_path and auto_file_name:
                path = Path(ckpt_path).parent / auto_file_name
                if path.exists():
                    with open(path, "rb") as f:
                        return pickle.load(f)  # noqa : S301
            return {}

        if isinstance(config, dict):
            return config

        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        print(suffix)

        if suffix in [".yaml", ".yml"]:
            with open(path) as f:
                return yaml.safe_load(f) or {}

        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. Use .yaml, .yml, or .pkl"
            )

    @classmethod
    def get_cls(cls):
        """Get the underlying model class."""
        raise NotImplementedError("Subclasses must implement `get_cls`.")

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        raise NotImplementedError("Subclasses must implement `get_datamodule_cls`.")

    @classmethod
    def get_test_dataset_from(cls, **kwargs):
        """
        Creates and returns D1 TimeSeries dataSet objects for testing.
        """
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates_v2,
            make_datasets_v2,
        )

        raw_data = data_with_covariates_v2()

        datasets_info = make_datasets_v2(raw_data, **kwargs)

        return {
            "train": datasets_info["training_dataset"],
            "predict": datasets_info["validation_dataset"],
        }

    def _build_model(self, metadata: dict, **kwargs):
        """Instantiates the model, either from a checkpoint or from config."""
        model_cls = self.get_cls()
        if self.ckpt_path:
            self.model = model_cls.load_from_checkpoint(
                self.ckpt_path, metadata=metadata, **kwargs
            )
        elif self.model_cfg:
            self.model = model_cls(**self.model_cfg, metadata=metadata)
        else:
            self.model = None

    def _build_datamodule(self, data: TimeSeries) -> LightningDataModule:
        """Constructs a DataModule from a D1 layer object."""
        if not self.datamodule_cfg:
            raise ValueError("`datamodule_cfg` must be provided to build a datamodule.")
        datamodule_cls = self.get_datamodule_cls()
        return datamodule_cls(data, **self.datamodule_cfg)

    def _load_dataloader(
        self, data: TimeSeries | LightningDataModule | DataLoader
    ) -> DataLoader:
        """Converts various data input types into a DataLoader for prediction."""
        if isinstance(data, TimeSeries):  # D1 Layer
            dm = self._build_datamodule(data)
            dm.setup(stage="predict")
            return dm.predict_dataloader()
        elif isinstance(data, LightningDataModule):  # D2 Layer
            data.setup(stage="predict")
            return data.predict_dataloader()
        elif isinstance(data, DataLoader):
            return data
        else:
            raise TypeError(
                f"Unsupported data type for prediction: {type(data).__name__}. "
                "Expected TimeSeriesDataSet, LightningDataModule, or DataLoader."
            )

    def _save_artifact(self, output_dir: Path):
        """Save all configuration artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "datamodule_cfg.pkl", "wb") as f:
            pickle.dump(self.datamodule_cfg, f)

        with open(output_dir / "model_cfg.pkl", "wb") as f:
            pickle.dump(self.model_cfg, f)

        if self.datamodule is not None and hasattr(self.datamodule, "metadata"):
            with open(output_dir / "metadata.pkl", "wb") as f:
                pickle.dump(self.datamodule.metadata, f)

    def fit(
        self,
        data: TimeSeries | LightningDataModule,
        # todo: we should create a base data_module for different data_modules
        save_ckpt: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        ckpt_kwargs: dict[str, Any] | None = None,
        **trainer_fit_kwargs,
    ):
        """
        Fit the model to the training data.

        Parameters
        ----------
        data : Union[TimeSeries, LightningDataModule]
            The data to fit on (D1 or D2 layer). This object is responsible
            for providing both training and validation data.
        save_ckpt : bool, default=True
            If True, save the best model checkpoint and the `datamodule_cfg`.
        ckpt_dir : Union[str, Path], default="checkpoints"
            Directory to save artifacts.
        ckpt_kwargs : dict, optional
            Keyword arguments passed to ``ModelCheckpoint``.
        **trainer_fit_kwargs :
            Additional keyword arguments passed to `trainer.fit()`.

        Returns
        -------
        Optional[Path]
            The path to the best model checkpoint if `save_ckpt=True`, else None.
        """
        if isinstance(data, TimeSeries):
            self.datamodule = self._build_datamodule(data)
        else:
            self.datamodule = data
        self.datamodule.setup(stage="fit")

        if self.model is None:
            if not self.model_cfg:
                raise RuntimeError(
                    "`model_cfg` must be provided to train from scratch."
                )
            metadata = self.datamodule.metadata
            self._build_model(metadata)

        callbacks = self.trainer_cfg.get("callbacks", []).copy()
        checkpoint_cb = None
        if save_ckpt:
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            default_ckpt_kwargs = {
                "dirpath": ckpt_dir,
                "filename": "best-{epoch}-{step}",
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
            }
            if ckpt_kwargs:
                default_ckpt_kwargs.update(ckpt_kwargs)
            checkpoint_cb = ModelCheckpoint(**default_ckpt_kwargs)
            callbacks.append(checkpoint_cb)
        trainer_init_cfg = self.trainer_cfg.copy()
        trainer_init_cfg.pop("callbacks", None)

        self.trainer = Trainer(**trainer_init_cfg, callbacks=callbacks)

        self.trainer.fit(self.model, datamodule=self.datamodule, **trainer_fit_kwargs)
        if save_ckpt and checkpoint_cb:
            best_model_path = Path(checkpoint_cb.best_model_path)
            self._save_artifact(best_model_path.parent)
            print(f"Artifacts saved in: {best_model_path.parent}")
            return best_model_path
        return None

    def predict(
        self,
        data: TimeSeries | LightningDataModule | DataLoader,
        output_dir: str | Path | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor] | None:
        """
        Generate predictions by wrapping the model's predict method.

        This method prepares the data by resolving it into a DataLoader and then
        delegates the prediction task to the underlying model's ``.predict()`` method.

        Parameters
        ----------
        data : Union[TimeSeries, LightningDataModule, DataLoader]
            The data to predict on (D1, D2, or DataLoader).
        **kwargs :
            Additional keyword arguments passed directly to the model's ``.predict()``
            method. This includes `mode`, `return_info`, `output_dir`, and any
            `trainer_kwargs`.

        Returns
        -------
        Union[Dict[str, torch.Tensor], None]
            A dictionary of prediction tensors, or `None` if `output_dir` is specified
            in `**kwargs`.
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not initialized. Provide `model_cfg` or `ckpt_path`."
            )

        dataloader = self._load_dataloader(data)
        predictions = self.model.predict(dataloader, **kwargs)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / "predictions.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(predictions, f)
            print(f"Predictions saved to {output_file}")
            return None

        return predictions
