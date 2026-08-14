from pathlib import Path
from typing import Any
import warnings

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import yaml


class _ArtifactRegistry:
    """Reads/writes artifacts.yaml."""

    @staticmethod
    def _read(registry_path: Path) -> dict[str, Any]:
        """Read the yaml file, returns ``{"artifacts": {}}`` if it doesn't exist yet."""
        if not registry_path.exists():
            return {"artifacts": {}}
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        return data if data is not None else {"artifacts": {}}

    @staticmethod
    def write(
        registry_path: Path, artifacts: dict[str, Any], overwrite: bool = False
    ) -> None:
        """Initial write.

        It is used by _save() to save scalers and other "static" artifacts
        before training starts. The static artifacts are those artifacts which never
        change during the training process - like the configs, datamodule.metadata,
        scalers etc. The non-static artifacts changes as the training/prediction
        process moves forward. Eg of this would be the model checkpoints - if we plan
        to save "best" model (any model that has better performance based on some
        metric)

        Parameters
        ----------
        registry_path : Path
           the path of the yaml file where we have to write the artifacts.
        artifacts : dict[str, Any]
           A dictionary of the artifacts we want to write.
           The Keys of the dictionary would be the "type" of artifact - like scaler,
           configs etc. And the value would be the actual object path to be written.
        overwrite: bool, default=False
            Whether to overwrite the artifact.yaml (if present) or not.
        """
        registry_path = Path(registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        if registry_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{registry_path} already exists. Pass `overwrite=True` to "
                    "_ArtifactRegistry.write() to replace it, or use .update() "
                    "if you only want to add/replace specific keys. You can also delete"
                    "the file if that is not needed."
                )
            warnings.warn(
                f"Overwriting existing {registry_path}. Any keys not present in "
                "the new `artifacts` dict will be lost. Use .update() instead if "
                "you want to preserve existing keys."
            )

        payload = {"artifacts": {k: str(v) for k, v in artifacts.items()}}
        with open(registry_path, "w") as f:
            yaml.safe_dump(payload, f)

    @staticmethod
    def update(registry_path: Path, artifacts: dict[str, Any]) -> None:
        """Merge-update specific keys.
        It is used by ArtifactRegistryCallback during
        training, and by _save() for the manual_checkpoint key. It will never overwrite
        the artifacts.yaml file, but just update an existing artifact entry.

        Parameters
        ----------
        registry_path : Path
             the path of the yaml file where we have to write the artifacts.
        artifacts : dict[str, Any]
            A dictionary where each key is the artifact type (eg.
            `best_model_checkpoint`) and the value is the path to set/replace
            for that key. Can contain one or more entries. Existing keys not
            present in this dict are left untouched.
        """
        registry_path = Path(registry_path)
        if not artifacts:
            return

        existing = _ArtifactRegistry._read(registry_path)
        existing.setdefault("artifacts", {})
        existing["artifacts"].update({k: str(v) for k, v in artifacts.items()})

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, "w") as f:
            yaml.safe_dump(existing, f)

    @staticmethod
    def get(registry_path: Path, key: str | None = None) -> dict[str, Any] | None:
        """Read one key, or the whole registry.

        Parameters
        -----------
        registry_path : Path
            the path of the yaml file from where we have to read the artifacts.
        key : str
            the key we want to know.

        Returns
        -------
        dict
            The dictionary if with the key as the key passed and value as the value
            read, or None with a warning if missing/file doesn't exist yet.
        key : str, default=None
            the key we want to know. If None, the entire artifacts
            dict is returned.
        """
        registry_path = Path(registry_path)
        if not registry_path.exists():
            warnings.warn(f"{registry_path} does not exist.")
            return None

        artifacts = _ArtifactRegistry._read(registry_path).get("artifacts", {})
        if key is None:
            return artifacts

        value = artifacts.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found in {registry_path}.")

        return {key: value}


class ArtifactRegistryCallback(Callback):
    """Called every time Lightning's ModelCheckpoint actually persists a file.
    Re-reads ckpt_cb.best_model_path / .last_model_path (always current,
    never stale -- Lightning deletes superseded files itself) and writes
    that into artifacts.yaml as the live truth, not a historical log.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        ckpt_cb = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None
        )
        if ckpt_cb is None:
            warnings.warn(
                "ArtifactRegistryCallback found no ModelCheckpoint among "
                "trainer.callbacks; skipping artifacts.yaml update."
            )
            return

        updates: dict[str, Any] = {}
        if ckpt_cb.best_model_path:
            updates["best_model_checkpoint"] = ckpt_cb.best_model_path
        if ckpt_cb.save_last and ckpt_cb.last_model_path:
            updates["last_model_checkpoint"] = ckpt_cb.last_model_path

        if updates:
            _ArtifactRegistry.update(self.registry_path, updates)
