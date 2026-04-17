########################################################################################
# Disclaimer: This baseclass is still work in progress and experimental, please
# use with care. This class is a basic skeleton of how the base classes may look like
# in the version-2.
########################################################################################


from typing import Any, Optional, Union
from warnings import warn

from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_forecasting.callbacks.predict import PredictCallback
from pytorch_forecasting.metrics import Metric, MultiLoss
from pytorch_forecasting.utils._classproperty import classproperty


class BaseModel(LightningModule):
    """Base model for time series forecasting.

    Parameters
    ----------
    loss : Descendants of ``pytorch_forecasting.metrics.Metric`` class
        Loss function to use for training.
    logging_metrics : Optional[List[nn.Module]], optional
        List of metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str, callable]], optional
        Optimizer to use for training.
        Can be a string ("adam", "adamw", "adagrad", "sgd", or any
        ``torch.optim`` class name), a callable returning an optimizer,
        or an instance of ``torch.optim.Optimizer``.
    optimizer_params : Optional[Dict], optional
        Parameters for the optimizer.
    lr_scheduler : Optional[str], optional
        Learning rate scheduler to use.
        Supported values: "reduce_lr_on_plateau", "step_lr",
        "cosine_annealing", "cosine_annealing_warm_restarts".
    lr_scheduler_params : Optional[Dict], optional
        Parameters for the learning rate scheduler.
    fine_tune_strategy : str, optional (default="freeze_backbone")
        Strategy used when fine-tuning a pretrained model:

        * ``"freeze_backbone"`` — freeze backbone parameters via
          :meth:`_freeze_backbone` so that only the head is updated.
        * ``"full"`` — fine-tune all layers with the same learning rate.
    pretrained_weights : str or None, optional
        Path to pretrained weights to load at initialisation time.
        Accepts a local ``.pt`` / ``.ckpt`` file path or a HuggingFace URI
        of the form ``hf://<org>/<repo>/<filename>``.
        Weights are loaded lazily: subclasses must call
        :meth:`_post_init_load_pretrained` at the **end** of their own
        ``__init__`` (after the architecture is fully built).

    Examples
    --------
    **Mode 1 — pretrain on large corpus, then fine-tune on target dataset:**

    >>> model = MyModel(loss=MAE(), metadata=dm_large.metadata)  # doctest: +SKIP
    >>> model.pretrain(dm_large, trainer_kwargs={"max_epochs": 20})  # doctest: +SKIP
    >>> from lightning import Trainer  # doctest: +SKIP
    >>> trainer = Trainer(max_epochs=5)  # doctest: +SKIP
    >>> trainer.fit(model, dm_target)  # backbone frozen automatically  # doctest: +SKIP

    **Mode 2 — cold-start: load community pretrained weights, skip pretrain():**

    >>> model = MyModel(  # doctest: +SKIP
    ...     loss=MAE(),
    ...     metadata=dm_target.metadata,
    ...     pretrained_weights="hf://org/my-pretrained-model/weights.pt",
    ... )
    >>> trainer = Trainer(max_epochs=5)  # doctest: +SKIP
    >>> trainer.fit(model, dm_target)  # fine-tunes from pretrained  # doctest: +SKIP

    **Mode 3 — train from scratch (backward compatible, no pretrain):**

    >>> model = MyModel(loss=MAE(), metadata=dm.metadata)  # doctest: +SKIP
    >>> trainer = Trainer(max_epochs=20)  # doctest: +SKIP
    >>> trainer.fit(model, dm)  # trains normally  # doctest: +SKIP
    """

    _OPTIMIZER_REGISTRY = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adagrad": torch.optim.Adagrad,
        "sgd": torch.optim.SGD,
    }

    _SCHEDULER_REGISTRY = {
        "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "step_lr": torch.optim.lr_scheduler.StepLR,
        "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_annealing_warm_restarts": (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ),
    }

    def __init__(
        self,
        loss: Metric,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        fine_tune_strategy: str = "freeze_backbone",
        pretrained_weights: str | None = None,
    ):
        super().__init__()
        self.loss = loss
        self.logging_metrics = nn.ModuleList(
            logging_metrics if logging_metrics is not None else []
        )
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = (
            lr_scheduler_params if lr_scheduler_params is not None else {}
        )
        self.fine_tune_strategy = fine_tune_strategy
        self._pretrained_weights_path = pretrained_weights
        self.model_name = self.__class__.__name__
        warn(
            f"The Model '{self.model_name}' is part of an experimental rework"
            "of the pytorch-forecasting model layer, scheduled for release with v2.0.0."
            " The API is not stable and may change without prior warning. "
            "This class is intended for beta testing and as a basic skeleton, "
            "but not for stable production use. "
            "Feedback and suggestions are very welcome in "
            "pytorch-forecasting issue 1736, "
            "https://github.com/sktime/pytorch-forecasting/issues/1736",
            UserWarning,
        )

    @classproperty
    def pkg(cls):
        """Package class for the model."""
        return cls._pkg()

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary containing input tensors

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing output tensors
        """
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def predict(
        self,
        dataloader: DataLoader,
        mode: str = "prediction",
        return_info: list[str] | None = None,
        mode_kwargs: dict[str, Any] = None,
        trainer_kwargs: dict[str, Any] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate predictions for new data using the `lightning.Trainer`.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader containing the data to predict on.
        mode : str
            The prediction mode ("prediction", "quantiles", or "raw").
        return_info : list[str], optional
            A list of additional information to return.
        mode_kwargs : dict[str, Any]
            Additional arguments for `to_prediction`/`to_quantiles`.
        trainer_kwargs: dict[str, Any]
            Additional arguments for `Trainer`.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of prediction results.
        """
        trainer_kwargs = trainer_kwargs or {}
        predict_callback = PredictCallback(
            mode=mode, return_info=return_info, mode_kwargs=mode_kwargs
        )

        callbacks = trainer_kwargs.get("callbacks", [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.append(predict_callback)
        trainer_kwargs["callbacks"] = callbacks

        trainer = Trainer(**trainer_kwargs)
        trainer.predict(self, dataloaders=dataloader)

        return predict_callback.result

    def to_prediction(self, out: dict[str, Any], **kwargs) -> torch.Tensor:
        """Converts raw model output to point forecasts."""
        # todo: add MultiLoss support
        try:
            out = self.loss.to_prediction(out["prediction"], **kwargs)
        except TypeError:  # in case passed kwargs do not exist
            out = self.loss.to_prediction(out["prediction"])
        return out

    def to_quantiles(self, out: dict[str, Any], **kwargs) -> torch.Tensor:
        """Converts raw model output to quantile forecasts."""
        # todo: add MultiLoss support
        try:
            out = self.loss.to_quantiles(out["prediction"], **kwargs)
        except TypeError:  # in case passed kwargs do not exist
            out = self.loss.to_quantiles(out["prediction"])
        return out

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input and target tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        STEP_OUTPUT
            Dictionary containing the loss and other metrics.
        """
        x, y = batch
        y_hat_dict = self(x)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="train")
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Validation step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input and target tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        STEP_OUTPUT
            Dictionary containing the loss and other metrics.
        """
        x, y = batch
        y_hat_dict = self(x)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="val")
        return {"val_loss": loss}

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Test step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input and target tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        STEP_OUTPUT
            Dictionary containing the loss and other metrics.
        """
        x, y = batch
        y_hat_dict = self(x)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="test")
        return {"test_loss": loss}

    def predict_step(
        self,
        batch: tuple[dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Prediction step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input tensors.
        batch_idx : int
            Index of the batch.
        dataloader_idx : int
            Index of the dataloader.

        Returns
        -------
        torch.Tensor
            Predicted output tensor.
        """
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizer and learning rate scheduler.

        Returns
        -------
        Dict
            Dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = self._get_optimizer()
        if self.lr_scheduler is not None:
            scheduler = self._get_scheduler(optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    },
                }
            else:
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def _get_optimizer(self) -> Optimizer:
        """
        Get the optimizer based on the specified optimizer name and parameters.

        Returns
        -------
        Optimizer
            The optimizer instance.
        """
        if callable(self.optimizer) and not isinstance(self.optimizer, str):
            return self.optimizer(self.parameters(), **self.optimizer_params)
        elif isinstance(self.optimizer, str):
            name = self.optimizer.lower()
            if name in self._OPTIMIZER_REGISTRY:
                opt_cls = self._OPTIMIZER_REGISTRY[name]
            elif hasattr(torch.optim, self.optimizer):
                opt_cls = getattr(torch.optim, self.optimizer)
            else:
                raise ValueError(f"Optimizer {self.optimizer} not supported.")
            return opt_cls(self.parameters(), **self.optimizer_params)
        elif isinstance(self.optimizer, Optimizer):
            return self.optimizer
        else:
            raise ValueError(
                "Optimizer must be a string, a callable, or "
                "an instance of torch.optim.Optimizer."
            )

    def _get_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get the lr scheduler based on the specified scheduler name and params.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler instance.
        """
        name = self.lr_scheduler.lower()
        if name in self._SCHEDULER_REGISTRY:
            sched_cls = self._SCHEDULER_REGISTRY[name]
            return sched_cls(optimizer, **self.lr_scheduler_params)
        else:
            raise ValueError(f"Scheduler {self.lr_scheduler} not supported.")

    def pretrain(
        self,
        datamodule,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> "BaseModel":
        """Pre-train the model on panel (global) data.

        Trains the model on a large dataset to learn shared patterns
        across multiple time series. After pretraining, calling
        ``Trainer.fit()`` will fine-tune the model without resetting
        the pretrained weights, as ``is_pretrained_`` is set to ``True``.

        Subclasses with special pretraining logic (e.g., backcast loss)
        should override ``_pretrain()`` instead of this method.

        Parameters
        ----------
        datamodule : LightningDataModule
            DataModule containing panel/global training data.
            Typically ``EncoderDecoderTimeSeriesDataModule`` or
            ``TslibDataModule`` with multiple time series instances.
        trainer_kwargs : dict, optional
            Additional keyword arguments passed to ``lightning.Trainer``.

        Returns
        -------
        self : reference to self

        Examples
        --------
        Three supported usage modes:

        **Mode 1 — pretrain on global data, then fine-tune on local data:**

        >>> model = NHiTS_v2(loss=MAE(), metadata=dm_large.metadata)  # doctest: +SKIP
        >>> model.pretrain(                                            # doctest: +SKIP
        ...     dm_large,                                              # doctest: +SKIP
        ...     trainer_kwargs={"max_epochs": 20},                    # doctest: +SKIP
        ... )
        >>> assert model.is_pretrained_                               # doctest: +SKIP
        >>> trainer = Trainer(max_epochs=10)                          # doctest: +SKIP
        >>> trainer.fit(model, dm_small)  # backbone frozen by default # doctest: +SKIP

        **Mode 2 — cold-start: load pretrained weights at init, skip pretrain():**

        >>> model = NHiTS_v2(                                         # doctest: +SKIP
        ...     loss=MAE(),                                           # doctest: +SKIP
        ...     metadata=dm_small.metadata,                           # doctest: +SKIP
        ...     pretrained_weights="pretrained.pt",                   # doctest: +SKIP
        ... )
        >>> assert model.is_pretrained_                               # doctest: +SKIP
        >>> trainer = Trainer(max_epochs=10)                          # doctest: +SKIP
        >>> trainer.fit(model, dm_small)  # fine-tune directly        # doctest: +SKIP

        **Mode 3 — train from scratch (backward compatible, no pretrain):**

        >>> model = NHiTS_v2(loss=MAE(), metadata=dm_small.metadata)  # doctest: +SKIP
        >>> trainer = Trainer(max_epochs=30)                          # doctest: +SKIP
        >>> trainer.fit(model, dm_small)                              # doctest: +SKIP
        """
        self._pretrain(datamodule, trainer_kwargs=trainer_kwargs)
        self.is_pretrained_ = True
        return self

    def _pretrain(
        self,
        datamodule,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Core pretraining logic, to be overridden by subclasses.

        Default behaviour: run ``Trainer.fit()`` on the given datamodule.
        Subclasses that need different loss weighting or freezing strategies
        during pretraining should override this method.

        Parameters
        ----------
        datamodule : LightningDataModule
            DataModule for pretraining.
        trainer_kwargs : dict, optional
            Additional keyword arguments passed to ``lightning.Trainer``.
        """
        trainer_kwargs = trainer_kwargs or {}
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(self, datamodule)

    def on_fit_start(self) -> None:
        """Lightning hook called at the start of ``Trainer.fit()``.

        When the model has been pretrained (``is_pretrained_ is True``) and
        ``fine_tune_strategy == "freeze_backbone"``, the backbone is frozen so
        that only the head parameters are updated during fine-tuning.
        """
        if getattr(self, "is_pretrained_", False):
            if self.fine_tune_strategy == "freeze_backbone":
                self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters, leaving the head trainable.

        Default behaviour: freeze all parameters in ``self.model`` if that
        attribute exists; otherwise freeze every parameter in the module.

        Subclasses with a clear backbone / head split should override this
        method to freeze only the backbone, keeping the head trainable.
        """
        target = getattr(self, "model", self)
        for param in target.parameters():
            param.requires_grad_(False)

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters.

        Call this after fine-tuning if you want to continue training all
        layers (e.g., for a second fine-tuning phase at a lower LR).
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def _post_init_load_pretrained(self) -> None:
        """Load pretrained weights stored by ``__init__`` if a path was given.

        Subclasses must call this at the **end** of their own ``__init__``
        (after the model architecture is fully built) to support the
        ``pretrained_weights`` parameter inherited from :class:`BaseModel`.

        Examples
        --------
        >>> class MyModel(BaseModel):  # doctest: +SKIP
        ...     def __init__(self, ..., **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.model = ...  # build architecture
        ...         self._post_init_load_pretrained()  # load weights if provided
        """
        if self._pretrained_weights_path is not None:
            self.load_pretrained_weights(self._pretrained_weights_path)

    def load_pretrained_weights(
        self,
        path: str,
        strict: bool = False,
    ) -> "BaseModel":
        """Load pretrained weights from a local checkpoint or HuggingFace hub.

        Allows skipping the ``pretrain()`` step by loading previously
        saved weights directly. Sets ``is_pretrained_`` to ``True``.

        Parameters
        ----------
        path : str
            Path to a checkpoint file (``.ckpt`` or ``.pt``), or a
            HuggingFace repo ID prefixed with ``"hf://"``
            (e.g., ``"hf://username/my-nhits-pretrained"``).
        strict : bool, optional (default=False)
            Whether to strictly enforce that the keys in ``state_dict``
            match the keys returned by this module's ``state_dict()``.
            ``False`` allows partial weight loading (useful when
            fine-tuning with a modified head).

        Returns
        -------
        self : reference to self

        Examples
        --------
        >>> model = MyModel(...)  # doctest: +SKIP
        >>> model.load_pretrained_weights("pretrained.ckpt")  # doctest: +SKIP
        >>> assert model.is_pretrained_  # doctest: +SKIP
        >>> model.load_pretrained_weights("hf://org/repo")  # doctest: +SKIP
        """
        resolved_path = self._resolve_checkpoint_path(path)
        checkpoint = torch.load(resolved_path, map_location="cpu")
        # lightning checkpoints store weights under "state_dict" key
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=strict)
        self.is_pretrained_ = True
        return self

    @staticmethod
    def _resolve_checkpoint_path(path: str) -> str:
        """Resolve a checkpoint path, downloading from HuggingFace if needed.

        Parameters
        ----------
        path : str
            Local file path or ``"hf://<repo_id>/<filename>"`` URI.

        Returns
        -------
        str
            Local file path ready for ``torch.load``.
        """
        if not path.startswith("hf://"):
            return path

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to load weights from the Hub. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        # hf://<repo_id>/<filename>  e.g. hf://myorg/nhits-pretrained/model.pt
        remainder = path[len("hf://") :]
        parts = remainder.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                "HuggingFace path must be 'hf://<owner>/<repo>/<filename>', "
                f"got: {path!r}"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def log_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str = "val"
    ) -> None:
        """
        Log additional metrics during training, validation, or testing.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted output tensor.
        y : torch.Tensor
            Target output tensor.
        prefix : str
            Prefix for the logged metrics (e.g., "train", "val", "test").
        """
        for metric in self.logging_metrics:
            metric_value = metric(y_hat, y)
            self.log(
                f"{prefix}_{metric.__class__.__name__}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
