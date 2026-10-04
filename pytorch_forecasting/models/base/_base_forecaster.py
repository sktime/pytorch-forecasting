from pathlib import Path
import pickle
from typing import Any, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.datamodule import LightningDataModule
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.models.base._base_object import _BasePtObject_v2


class BaseForecaster(_BasePtObject_v2):
    """
    Base forecaster class acting as a high-level wrapper for the Lightning workflow.

    Model parameters are flattened into the concrete subclass's ``__init__``; the
    model is built inside ``fit`` from the datamodule's ``metadata``. The class
    manages model, datamodule and trainer, and provides streamlined ``fit`` and
    ``predict`` methods.

    Parameters
    ----------
    trainer : lightning.Trainer, optional
        Trainer used by ``fit`` when none is passed there. Defaults to ``Trainer()``.
    datamodule : LightningDataModule, optional
        A configured but data-less datamodule; ``fit`` attaches the data with
        ``with_data``. Defaults to ``get_datamodule_cls()()``, or to the
        ``datamodule_cfg.pkl`` saved next to ``ckpt_path`` if loading.
    ckpt_path : Union[str, Path], optional
        Path to the checkpoint from which to load the model. Defaults to None.
    """

    def __init__(
        self,
        trainer: Trainer | None = None,
        datamodule: Any = None,
        ckpt_path: str | Path | None = None,
    ):
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = ckpt_path
        super().__init__()

        self._loaded_datamodule_cfg = self._load_config(
            None, ckpt_path=self.ckpt_path, auto_file_name="datamodule_cfg.pkl"
        )
        self.metadata = self._load_config(
            None, ckpt_path=self.ckpt_path, auto_file_name="metadata.pkl"
        )
        self._loaded_model_cfg = self._load_config(
            None, ckpt_path=self.ckpt_path, auto_file_name="model_cfg.pkl"
        )

        self.model = None
        self.trainer_ = None
        self.datamodule_ = None
        self._is_fitted = False
        if self.ckpt_path:
            self._build_model(metadata=self.metadata)

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

        if suffix in [".yaml", ".yml"]:
            with open(path) as f:
                return yaml.safe_load(f) or {}

        elif suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301
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

    def get_model_params(self) -> dict[str, Any]:
        """Return the kwargs for ``get_cls()``, excluding ``metadata``."""
        raise NotImplementedError("Subclasses must implement `get_model_params`.")

    @property
    def model_cfg(self) -> dict[str, Any]:
        """Model kwargs, as passed to ``get_cls()``."""
        return self.get_model_params()

    @property
    def datamodule_cfg(self) -> dict[str, Any]:
        """Constructor kwargs of the datamodule, fitted one preferred."""
        dm = self.datamodule_ if self.datamodule_ is not None else self.datamodule
        if dm is None:
            return self._loaded_datamodule_cfg
        if hasattr(dm, "_init_kwargs"):
            return dm._init_kwargs
        return dict(getattr(dm, "hparams", {}))

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

    def _check_is_fitted(self, method_name: str | None = None):
        """Raise ``RuntimeError`` if ``fit`` has not been called."""
        if not self._is_fitted:
            caller = f"`{method_name}` " if method_name else ""
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; {caller}requires "
                "`fit` to be called first."
            )

    def _build_model(self, metadata: dict):
        """Instantiates the model, either from a checkpoint or from config."""
        model_cls = self.get_cls()
        if self.ckpt_path:
            self.model = model_cls.load_from_checkpoint(
                self.ckpt_path, metadata=metadata, **self._loaded_model_cfg
            )
        else:
            self.model = model_cls(**self.model_cfg, metadata=metadata)

    def _build_datamodule(self, data: TimeSeries) -> LightningDataModule:
        """Attach ``data`` to the configured datamodule, with no fitted transforms."""
        dm = self.datamodule
        if dm is None:
            dm = self.get_datamodule_cls()(**self._loaded_datamodule_cfg)
        return dm.with_data(data)

    def _resolve_trainer(self, trainer: Trainer | None) -> Trainer:
        """``fit``'s trainer > ``__init__``'s trainer > default ``Trainer()``."""
        if trainer is not None:
            return trainer
        if self.trainer is not None:
            return self.trainer
        return Trainer()

    def _load_dataloader(self, data: TimeSeries) -> DataLoader:
        """Build the prediction dataloader, with the transforms fitted in ``fit``."""
        if self._is_fitted:
            dm = self.datamodule_.with_data(data)
        else:
            dm = self._build_datamodule(data)
        dm.setup(stage="predict")
        return dm.predict_dataloader()

    @staticmethod
    def _to_timeseries(
        y: torch.Tensor | list[torch.Tensor],
        windows: list | None,
        metadata: Any | None,
    ) -> TimeSeries:
        """Move the pred tensors to ``TimeSeries``.

        Parameters
        ----------
        y : torch.Tensor or list of torch.Tensor
            Model output tensor.
        windows : list of tuple, optional
            One entry per window, ``(series_idx, start_idx, encoder_length,
            prediction_length)``, as recorded by the prediction dataset. Used
            to label each row with its source series and time position. If
            ``None`` or its length differs from ``n_windows``, each window is
            labelled as its own series with a horizon-local time index.
        metadata : TimeSeriesMetadata or dict, optional
            Schema of the input data.
            We read only ``metadata["cols"]["y"]`` is read, to
            name the target columns. Pass ``None`` when the last axis does not
            hold targets, e.g. in quantile mode, so columns fall back to
            ``y0, y1, ...``.

        Returns
        -------
        TimeSeries
            ``n_windows * prediction_length`` rows, one per forecast step, with
            ``metadata.is_prediction=True`` and columns:

            * ``_series`` : index of the source series (``series_idx`` of the
              window), also the ``group`` of the ``TimeSeries``.
            * ``_time_idx`` : position in that series,
            * one column per output, the targets of the ``TimeSeries``. Named
              after the targets in point mode when ``metadata`` is given, else
              ``y0, y1, ...``. With several targets and quantile outputs the
              columns are target-major, i.e. all quantiles of target 0, then
              all quantiles of target 1.

            Values are in the space of the fitted target normalizer.
        """
        if isinstance(y, (list, tuple)):
            y = torch.stack(list(y), dim=2)
        n_windows, pred_len = y.shape[:2]

        if windows is None or len(windows) != n_windows:
            series = np.repeat(np.arange(n_windows), pred_len)
            t = np.tile(np.arange(pred_len), n_windows)
        else:
            series = np.repeat([w[0] for w in windows], pred_len)
            t = np.concatenate(
                [np.arange(s + enc, s + enc + pred_len) for _, s, enc, _ in windows]
            )

        return TimeSeries.from_tensors(
            {"y": y.reshape(n_windows * pred_len, -1), "t": torch.as_tensor(t)},
            metadata=metadata,
            groups=series,
        )

    def _save_artifact(self, output_dir: Path):
        """Save all configuration artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "datamodule_cfg.pkl", "wb") as f:
            pickle.dump(self.datamodule_cfg, f)

        with open(output_dir / "model_cfg.pkl", "wb") as f:
            pickle.dump(self.model_cfg, f)

        if self.datamodule_ is not None and hasattr(self.datamodule_, "metadata"):
            with open(output_dir / "metadata.pkl", "wb") as f:
                pickle.dump(self.datamodule_.metadata, f)

    def fit(
        self,
        data: TimeSeries,
        trainer: Trainer | None = None,
        save_ckpt: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        ckpt_kwargs: dict[str, Any] | None = None,
        **trainer_fit_kwargs,
    ):
        """
        Fit the model to the training data.

        Parameters
        ----------
        data : TimeSeries
            The data to fit on. The datamodule splits it into training and
            validation data.
        trainer : lightning.Trainer, optional
            Overrides the trainer given to ``__init__`` for this call only.
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
        self.datamodule_ = self._build_datamodule(data)

        # the model is built only here, because only now are its input
        # shapes known - they come from the metadata of the data module
        metadata = self.datamodule_.metadata
        self._build_model(metadata)

        self.trainer_ = self._resolve_trainer(trainer)
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
            self.trainer_.callbacks.append(checkpoint_cb)

        self.trainer_.fit(self.model, datamodule=self.datamodule_, **trainer_fit_kwargs)
        self._is_fitted = True
        if save_ckpt and checkpoint_cb:
            best_model_path = Path(checkpoint_cb.best_model_path)
            self._save_artifact(best_model_path.parent)
            print(f"Artifacts saved in: {best_model_path.parent}")
            return best_model_path
        return None

    def predict(
        self,
        data: TimeSeries,
        mode: str = "prediction",
        return_info: list[str] | None = None,
        output_dir: str | Path | None = None,
        **kwargs,
    ) -> TimeSeries | dict[str, torch.Tensor] | None:
        """
        Generate predictions by wrapping the model's predict method.

        This method prepares the data by resolving it into a DataLoader and then
        delegates the prediction task to the underlying model's ``.predict()`` method.

        Parameters
        ----------
        data : TimeSeries
            The data to predict on. If the datamodule has a target normalizer
            or scalers, the data is scaled with the ones fitted in ``fit``.
        mode : str
            The prediction mode ("prediction", "quantiles", or "raw").
        return_info : list of str, optional
            Extra keys to return next to the prediction, e.g. ``"x"``, ``"y"``,
            ``"index"``. Forces the dict return form.
        output_dir : str or Path, optional
            If given, the result is pickled to ``predictions.pkl`` there and
            ``None`` is returned.
        **kwargs :
            Passed on to the model's ``.predict()``, e.g. ``mode_kwargs`` and
            ``trainer_kwargs``.

        Returns
        -------
        TimeSeries or dict of str to torch.Tensor or None
            For ``mode="prediction"`` and ``"quantiles"`` without ``return_info``,
            a :class:`TimeSeries` with ``metadata.is_prediction=True`` and one row
            per forecast step. ``_series`` is the source series of the window and
            ``_time_idx`` the position in that series; windows that overlap repeat
            a ``_time_idx`` within a ``_series``. Point mode names the target
            columns after the targets; quantile mode uses ``y0, y1, ...``, one
            per quantile. For ``mode="raw"`` or when ``return_info`` is given,
            the model's dict. ``None`` if ``output_dir`` is given.

            Values are in the space of the fitted target normalizer; no inverse
            transform is applied yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not initialized. Call `fit` or pass `ckpt_path`."
            )

        dataloader = self._load_dataloader(data)
        predictions = self.model.predict(
            dataloader, mode=mode, return_info=return_info, **kwargs
        )
        if mode != "raw" and not return_info:
            predictions = self._to_timeseries(
                predictions["prediction"],
                windows=getattr(dataloader.dataset, "windows", None),
                metadata=data.get_metadata() if mode == "prediction" else None,
            )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / "predictions.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(predictions, f)
            print(f"Predictions saved to {output_file}")
            return None

        return predictions
