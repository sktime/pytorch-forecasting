from pathlib import Path
import pickle
from typing import Any, Optional, Union
import warnings

from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.core.datamodule import LightningDataModule
import torch
from torch.utils.data import DataLoader
import yaml

from pytorch_forecasting.callbacks.artifact_registry import (
    ArtifactRegistryCallback,
    _ArtifactRegistry,
)
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

    _CFG_KEYS = ("model_cfg", "datamodule_cfg", "trainer_cfg")
    _DATAMODULE_KEYS = ("scalers", "target_normalizer")

    def __init__(
        self,
        model_cfg: dict[str, Any] | str | Path | None = None,
        trainer_cfg: dict[str, Any] | str | Path | None = None,
        datamodule_cfg: dict[str, Any] | str | Path | None = None,
        ckpt_path: str | Path | None = None,
    ):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.model_cfg = self._load_config(model_cfg)
        self.datamodule_cfg = self._load_config(datamodule_cfg)
        self.trainer_cfg = self._load_config(trainer_cfg)

        if not self.model_cfg and not self.datamodule_cfg and not self.trainer_cfg:
            warnings.warn(
                "No configs or `ckpt_path` were provided -- this pkg has "
                "nothing to build from yet. If you're trying to restore a "
                "previously saved pkg, use `Base_pkg.load(ckpt_dir)` "
                "instead of the constructor."
            )

        self.metadata = {}
        self.model = None
        self.trainer = None
        self.datamodule = None

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

    def _init_model_from_cfg(self, metadata: dict):
        """Construct a fresh model from ``self.model_cfg``."""
        if not self.model_cfg:
            raise RuntimeError("`model_cfg` must be provided to train from scratch.")
        model_cls = self.get_cls()
        return model_cls(**self.model_cfg, metadata=metadata)

    def _load_model_from_checkpoint(self, ckpt_path: str | Path, metadata: dict):
        """Deserialize model weights from a checkpoint."""
        model_cls = self.get_cls()
        return model_cls.load_from_checkpoint(
            ckpt_path, metadata=metadata, **self.model_cfg
        )

    def _build_datamodule(self, data: TimeSeries) -> LightningDataModule:
        """Constructs a DataModule from a D1 layer object."""
        if not self.datamodule_cfg:
            raise ValueError("`datamodule_cfg` must be provided to build a datamodule.")
        datamodule_cls = self.get_datamodule_cls()
        dm = datamodule_cls(data, **self.datamodule_cfg)
        if hasattr(self, "_pending_dm_artifacts"):
            dm.load_artifacts(self._pending_dm_artifacts)
            self._pending_dm_artifacts = {}

        return dm

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

    def _save(
        self,
        ckpt_dir: Path,
        model_ckpt_kwargs: dict[str, Any] | None = None,
        exclude: list[str] | None = None,
        overwrite: bool = False,
    ) -> tuple[Path, list[Callback]]:
        """Private. Only called from fit().

        Writes cfgs directly (pkg owns these). Delegates scalers/target_normalizer
        /metadata to datamodule.save_artifacts(). Fills artifacts.yaml with
        everything written so far. Builds (but does not run) the ModelCheckpoint +
        ArtifactRegistryCallback pair that will write the model-checkpoint keys
        later, during trainer.fit().

        Save contract
        ------------------
        - Saving is never a single synchronous event. Eager artifacts (cfgs, scalers,
          target_normalizer, datamodule metadata) are written immediately, before a
          Trainer exists. Model weights are written later, on Lightning's own
          schedule, via ModelCheckpoint + ArtifactRegistryCallback.
        - `_save()` is private. It is only ever called from `fit()`, which is the
          only context where "save" has an unambiguous meaning (we're either about
          to train, or training already happened in this same call). There is no
          public standalone save.
        - `best_model_checkpoint` / `last_model_checkpoint` keys in artifacts.yaml
          are owned exclusively by ArtifactRegistryCallback. Nothing else writes them.

        Parameters
        ----------
        ckpt_dir : Path
            directory where the chekcpoints are saved
        model_ckpt_kwargs : dict[str, Any] | None, defualt = None
            kwargs for ModelCheckpoint
        exclude : list[str], defualt = None
            the artifacts we want to exclude while saving
        overwrite : bool, default=False
            Whether to overwrite the `ckpt_dir` (if present) or not.

        Returns
        -------
        registry_path : Path
            path of the artifacts.yaml
        callbacks : list[Callback]
            Empty if "model_checkpoint" in exclude.
        """
        exclude = exclude or []
        ckpt_dir = Path(ckpt_dir)
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{ckpt_dir} is not empty. Pass `overwrite=True` to "
                    "`fit()` to replace its contents, or delete the directory "
                    "if it is no longer needed."
                )
            warnings.warn(
                f"Overwriting existing {ckpt_dir}. Any files not regenerated "
                "in this run will be lost."
            )

        configs_dir = ckpt_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        cfg_values = {
            "model_cfg": self.model_cfg,
            "datamodule_cfg": self.datamodule_cfg,
            "trainer_cfg": self.trainer_cfg,
        }
        for name in self._CFG_KEYS:
            path = configs_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(cfg_values[name], f)
            artifacts[name] = path

        if self.datamodule is not None:
            dm_artifacts = self.datamodule.save_artifacts(ckpt_dir, exclude=exclude)
            if not dm_artifacts:
                warnings.warn(
                    "Datamodule had nothing to save (no scalers, target_normalizer, "
                    "or metadata found)."
                )
            else:
                artifacts.update(dm_artifacts)

        registry_path = ckpt_dir / "artifacts.yaml"
        _ArtifactRegistry.write(registry_path, artifacts, overwrite=True)

        if "model_checkpoint" in exclude:
            return registry_path, []

        default_ckpt_kwargs = {
            "dirpath": ckpt_dir / "checkpoints",
            "filename": "best-{epoch}-{step}",
            "save_top_k": 1,
            "monitor": "val_loss",
            "mode": "min",
            "save_last": True,
        }
        if model_ckpt_kwargs:
            default_ckpt_kwargs.update(model_ckpt_kwargs)
        checkpoint_cb = ModelCheckpoint(**default_ckpt_kwargs)
        registry_cb = ArtifactRegistryCallback(registry_path)

        return registry_path, [checkpoint_cb, registry_cb]

    @classmethod
    def load(
        cls,
        ckpt_dir: str | Path,
        skip: list[str] | None = None,
    ) -> None:
        """Public, single entry point for loading. Reads artifacts.yaml and
        delegates each artifact to whichever layer owns it.

        Load contract
        ------------------
        - Loading is a single synchronous event, unlike saving.
        - `load()` is the only entry point, and it is public.
        - `load()` reads artifacts.yaml once and delegates each key to whichever layer
          owns it: cfgs are loaded directly by `pkg` (same layer that wrote them);
          scalers / target_normalizer / datamodule_metadata are handed to
          `datamodule.load_artifacts()`; `best_model_checkpoint` is handed to the
          model class's `load_from_checkpoint()`. `pkg` never deserializes an artifact
          it doesn't itself own.
        - Ordering is load-bearing: datamodule metadata must be loaded and the
          datamodule reconstructed *before* `_build_model()` runs, since model
          construction takes metadata as an argument. `load()` is responsible for
          this ordering -- the owning layers are not responsible for sequencing
          themselves correctly relative to each other.
        - `skip` lets the user exclude specific keys from artifacts.yaml at load time,
          independent of whatever `exclude` was passed at save time -- these are two
          separate controls over two separate moments, not the same flag reused.
        - A key missing from artifacts.yaml (because it was excluded at save time, or
          never existed -- e.g. no scalers were ever configured) is not an error.
          `load()` only acts on keys that are present; it never raises just because an
          optional artifact wasn't there to begin with.
        - `load()` never writes to artifacts.yaml. It only reads. All registry writes
          belong to `_save()` and `ArtifactRegistryCallback`, on the saving side.

        Parameters
        ----------
        ckpt_dir : Path
            the directory we need to load from
        skip: list[str], defualt = []
            list of artifacts we dont want to load

        Returns
        -------
        Base_pkg
            A fully-formed instance with all restored state.

        Raises
        ------
        FileNotFoundError
            If ``artifacts.yaml`` is not found in ``ckpt_dir``.
        """
        skip = skip or []
        ckpt_dir = Path(ckpt_dir)
        registry_path = ckpt_dir / "artifacts.yaml"

        artifacts = _ArtifactRegistry.get(registry_path)
        if artifacts is None:
            raise FileNotFoundError(
                f"No artifacts.yaml found at {registry_path}. Nothing to load."
            )
        artifacts = {k: v for k, v in artifacts.items() if k not in skip}

        obj = cls.__new__(cls)

        obj._init_empty_state()
        obj._populate_from_artifacts(artifacts)

        return obj

    def _init_empty_state(self) -> None:
        """Set all instance attributes to safe empty defaults."""
        self.ckpt_path = None
        self.model_cfg = {}
        self.datamodule_cfg = {}
        self.trainer_cfg = {}
        self.metadata = {}
        self.model = None
        self.trainer = None
        self.datamodule = None
        # Pending artifacts to inject when datamodule is created with data
        self._pending_dm_artifacts = {}

    def _populate_from_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Populate instance state from a parsed artifacts dictionary."""
        self._load_cfgs_from_artifacts(artifacts)
        self._load_metadata_from_artifacts(artifacts)
        self._init_trainer_from_cfg()
        self._store_pending_dm_artifacts(artifacts)
        self._load_model_from_artifacts(artifacts)

    def _load_cfgs_from_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Load configuration dictionaries from artifacts."""
        for name in self._CFG_KEYS:
            path = artifacts.get(name)
            if path is None:
                continue
            with open(path, "rb") as f:
                setattr(self, name, pickle.load(f))  # noqa: S301

    def _load_metadata_from_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Load metadata from datamodule."""
        metadata_path = artifacts.get("datamodule_metadata")
        if metadata_path is not None:
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)  # noqa: S301

    def _init_trainer_from_cfg(self) -> None:
        """Create Trainer instance from loaded config."""
        if not self.trainer_cfg:
            return
        trainer_init_cfg = self.trainer_cfg.copy()
        callbacks = trainer_init_cfg.pop("callbacks", [])

        self.trainer = Trainer(**trainer_init_cfg, callbacks=callbacks)

    def _store_pending_dm_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Store datamodule artifact paths for later injection."""
        self._pending_dm_artifacts = {
            k: v for k, v in artifacts.items() if k in self._DATAMODULE_KEYS
        }

    def _load_model_from_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Load model weights from checkpoint."""
        ckpt_path = artifacts.get("best_model_checkpoint")
        if ckpt_path is None:
            warnings.warn(
                "No 'best_model_checkpoint' found in artifacts.yaml -- model "
                "weights were not loaded."
            )
            return

        self.ckpt_path = Path(ckpt_path)
        self.model = self._load_model_from_checkpoint(
            self.ckpt_path, metadata=self.metadata
        )

    def fit(
        self,
        data: TimeSeries | LightningDataModule,
        # todo: we should create a base data_module for different data_modules
        ckpt_dir: str | Path | None = None,
        ckpt_kwargs: dict[str, Any] | None = None,
        exclude: list[str] | None = None,
        overwrite: bool = False,
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
        exclude : list of str, optional
            Artifacts to exclude from saving.
        overwrite : bool, default=False
            Whether to overwrite the `ckpt_dir` (if present) or not.
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
            # If user passed a datamodule directly, still send pending artifacts if any
            if self._pending_dm_artifacts and hasattr(
                self.datamodule, "load_artifacts"
            ):
                self.datamodule.load_artifacts(self._pending_dm_artifacts)
                self._pending_dm_artifacts = {}

        self.datamodule.setup(stage="fit")

        if hasattr(self.datamodule, "metadata"):
            self.metadata = self.datamodule.metadata

        if self.model is None:
            self.model = self._init_model_from_cfg(self.metadata)

        save_callbacks = []
        registry_path = None
        if ckpt_dir:
            registry_path, save_callbacks = self._save(
                Path(ckpt_dir),
                model_ckpt_kwargs=ckpt_kwargs,
                exclude=exclude,
                overwrite=overwrite,
            )

        # Use existing trainer if it was created during load(), otherwise create new
        if self.trainer is None:
            user_callbacks = self.trainer_cfg.get("callbacks", []).copy()
            trainer_init_cfg = self.trainer_cfg.copy()
            trainer_init_cfg.pop("callbacks", None)
            self.trainer = Trainer(
                **trainer_init_cfg, callbacks=[*user_callbacks, *save_callbacks]
            )
        else:
            # Add save callbacks to existing trainer (from load)
            self.trainer.callbacks.extend(save_callbacks)

        self.trainer.fit(self.model, datamodule=self.datamodule, **trainer_fit_kwargs)
        if registry_path is None or not save_callbacks:
            return None

        best = _ArtifactRegistry.get(registry_path, "best_model_checkpoint")
        if best is None:
            return None
        print(f"Artifacts saved in: {Path(registry_path).parent}")
        return Path(best["best_model_checkpoint"])

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
