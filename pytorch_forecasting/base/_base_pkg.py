from pathlib import Path
import pickle
from typing import Any, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.datamodule import LightningDataModule
import torch
from torch.utils.data import DataLoader

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class Base_pkg(_BasePtForecasterV2):
    """
    Base model package class acting as a high-level wrapper for the Lightning workflow.

    This class simplifies the user experience by managing model, datamodule, and trainer
    configurations, and providing streamlined `fit` and `predict` methods.

    Parameters
    ----------
    model_cfg : dict, optional
        Model configs for the initialisation of the model. Required if not loading
        from a checkpoint. Defaults to {}.
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
        model_cfg: Optional[dict[str, Any]] = None,
        trainer_cfg: Optional[dict[str, Any]] = None,
        datamodule_cfg: Optional[Union[dict[str, Any], str, Path]] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
    ):
        self.model_cfg = model_cfg or {}
        self.trainer_cfg = trainer_cfg or {}
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

        if isinstance(datamodule_cfg, (str, Path)):
            with open(datamodule_cfg, "rb") as f:
                self.datamodule_cfg = pickle.load(f)  # noqa : S301
        else:
            self.datamodule_cfg = datamodule_cfg or {}

        self.model = None
        self.trainer = None
        self.datamodule = None

        self._build_model()

    @classmethod
    def get_cls(cls):
        """Get the underlying model class."""
        raise NotImplementedError("Subclasses must implement `get_cls`.")

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        raise NotImplementedError("Subclasses must implement `get_datamodule_cls`.")

    def _build_model(self):
        """Instantiates the model, either from a checkpoint or from config."""
        model_cls = self.get_cls()
        if self.ckpt_path:
            self.model = model_cls.load_from_checkpoint(self.ckpt_path)
        elif self.model_cfg:
            self.model = model_cls(**self.model_cfg)
        else:
            self.model = None

    def _build_datamodule(self, data: TimeSeries) -> LightningDataModule:
        """Constructs a DataModule from a D1 layer object."""
        if not self.datamodule_cfg:
            raise ValueError("`datamodule_cfg` must be provided to build a datamodule.")
        datamodule_cls = self.get_datamodule_cls()
        return datamodule_cls(data, **self.datamodule_cfg)

    def _load_dataloader(
        self, data: Union[TimeSeries, LightningDataModule, DataLoader]
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

    def fit(
        self,
        train_data: Union[TimeSeries, LightningDataModule],
        # todo: we should create a base data_module for different data_modules
        val_data: Optional[Union[TimeSeries, LightningDataModule]] = None,
        save_ckpt: bool = True,
        ckpt_dir: Union[str, Path] = "checkpoints",
        **trainer_fit_kwargs,
    ):
        """
        Fit the model to the training data.

        Parameters
        ----------
        train_data : Union[TimeSeries, LightningDataModule]
            Training data (D1 or D2 layer).
        val_data : Union[TimeSeries, LightningDataModule], optional
            Validation data.
        save_ckpt : bool, default=True
            If True, save the best model checkpoint and the `datamodule_cfg`.
        ckpt_dir : Union[str, Path], default="checkpoints"
            Directory to save artifacts.
        **trainer_fit_kwargs :
            Additional keyword arguments passed to `trainer.fit()`.

        Returns
        -------
        Optional[Path]
            The path to the best model checkpoint if `save_ckpt=True`, else None.
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not initialized. Provide `model_cfg` or `ckpt_path`."
            )

        if isinstance(train_data, TimeSeries):
            self.datamodule = self._build_datamodule(train_data)
        else:
            self.datamodule = train_data

        callbacks = self.trainer_cfg.get("callbacks", [])
        if save_ckpt:
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_cb = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-{epoch}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_cb)

        self.trainer = Trainer(**self.trainer_cfg, callbacks=callbacks)
        self.trainer.fit(self.model, datamodule=self.datamodule, **trainer_fit_kwargs)

        if save_ckpt and checkpoint_cb:
            best_model_path = Path(checkpoint_cb.best_model_path)
            dm_cfg_path = best_model_path.parent / "datamodule_cfg.pkl"
            with open(dm_cfg_path, "wb") as f:
                pickle.dump(self.datamodule_cfg, f)
            print(f"Best model saved to: {best_model_path}")
            print(f"DataModule config saved to: {dm_cfg_path}")
            return best_model_path
        return None

    def predict(
        self,
        data: Union[TimeSeries, LightningDataModule, DataLoader],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Union[dict[str, torch.Tensor], None]:
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

        return self.model.predict(dataloader, **kwargs)
