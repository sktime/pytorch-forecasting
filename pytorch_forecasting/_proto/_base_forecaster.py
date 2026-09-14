"""
``BaseForecaster`` - prototype for the v2 redesign.
"""

from typing import Any

from lightning import Trainer
import numpy as np

from pytorch_forecasting._proto._timeseries_datatype import TimeSeries_datatype
from pytorch_forecasting.base._base_pkg import Base_pkg


class BaseForecaster(Base_pkg):
    """Base class for forecasters - the user facing model object.

    A forecaster owns a model, a data module and a trainer, but the user does
    not construct any of them. Concrete forecasters flatten their model
    parameters into ``__init__`` and return them from ``get_model_params``.

    Parameters
    ----------
    trainer : lightning.Trainer, optional, default=None
        Trainer to fit with. May also be given to ``fit``, or built there from
        keyword arguments. A trainer given to ``fit`` takes precedence.
    datamodule : LightningDataModule, optional, default=None
        Data module to use, configured but without data - the data is supplied
        by ``fit``. If not given, ``get_datamodule_cls()`` is used with its
        default settings.

    Attributes
    ----------
    model_ : LightningModule
        The fitted model. Only present after ``fit``.
    datamodule_ : LightningDataModule
        Data module holding the training data. Only present after ``fit``.
    trainer_ : lightning.Trainer
        The trainer used for fitting. Only present after ``fit``.
    """

    _tags = {
        "object_type": "forecaster_pytorch_v2_proto",
    }

    def __init__(self, trainer: Trainer | None = None, datamodule: Any = None):
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = None
        self._is_fitted = False

    @property
    def model_cfg(self) -> dict[str, Any]:
        """Parameters passed on to the model, as a dict."""
        return self.get_model_params()

    @property
    def datamodule_cfg(self) -> dict[str, Any]:
        """Parameters the data module was configured with, as a dict.

        Read from the constructor arguments the data module recorded, falling
        back to ``hparams`` for modules that save their hyperparameters the
        lightning way instead.
        """
        datamodule = getattr(self, "datamodule_", None) or self.datamodule
        if datamodule is None:
            return {}
        if hasattr(datamodule, "_init_kwargs"):
            return dict(datamodule._init_kwargs)
        return dict(getattr(datamodule, "hparams", {}) or {})

    def get_model_params(self) -> dict[str, Any]:
        """Return the parameters to construct the model with.

        Returns
        -------
        dict
            keyword arguments for ``get_cls()``, excluding ``metadata``, which
            is taken from the data module in ``fit``.
        """
        raise NotImplementedError("concrete forecasters must implement this method")

    def _resolve_datamodule(self, data: TimeSeries_datatype):
        """Attach ``data`` to the configured data module."""
        datamodule = self.datamodule
        if datamodule is None:
            datamodule = self.get_datamodule_cls()()
        return datamodule.with_data(data)

    def _resolve_trainer(self, trainer) -> Trainer:
        """Pick the trainer to fit with.

        A trainer given to ``fit`` wins over one given to ``__init__``. If
        neither is given, a default ``Trainer`` is used.
        """
        trainer = trainer if trainer is not None else self.trainer
        if trainer is None:
            return Trainer()
        return trainer

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"this {type(self).__name__} is not fitted yet - "
                "call `fit` before `predict`"
            )

    def fit(
        self,
        data: TimeSeries_datatype,
        trainer: Trainer | None = None,
    ) -> "BaseForecaster":
        """Fit the forecaster to data.

        Parameters
        ----------
        data : TimeSeries_datatype
            Data to fit on. It is handed to the data module, which splits it
            into training and validation data.
        trainer : lightning.Trainer, optional, default=None
            Trainer to fit with. Takes precedence over the one given to
            ``__init__``. If neither is given, a default ``Trainer`` is used.

        Returns
        -------
        self : reference to self
        """
        self.datamodule_ = self._resolve_datamodule(data)

        # the model is built only here, because only now are its input shapes
        # known - they come from the metadata of the data module
        self.model_ = self.get_cls()(
            **self.model_cfg, metadata=self.datamodule_.metadata
        )
        self.trainer_ = self._resolve_trainer(trainer)
        self.trainer_.fit(self.model_, datamodule=self.datamodule_)

        self._is_fitted = True
        return self

    def predict(
        self,
        data: TimeSeries_datatype,
        mode: str = "prediction",
        **kwargs,
    ) -> TimeSeries_datatype:
        """Forecast for data.

        Parameters
        ----------
        data : TimeSeries_datatype
            Data to predict for. It is scaled with the transforms fitted in
            ``fit``, not with its own statistics.
        mode : str, default="prediction"
            One of ``"prediction"``, ``"quantiles"``, or ``"raw"``.
        **kwargs
            Passed on to ``model.predict``.

        Returns
        -------
        TimeSeries_datatype
            The forecasts, with ``metadata.is_prediction`` set to True. Use
            ``to_pandas`` to get them as a data frame.
        """
        self._check_is_fitted()

        datamodule = self.datamodule_.with_data(data)
        datamodule.setup(stage="predict")

        out = self.model_.predict(datamodule.predict_dataloader(), mode=mode, **kwargs)
        return self._to_timeseries(out, datamodule)

    def _to_timeseries(self, out: dict, datamodule) -> TimeSeries_datatype:
        """Wrap raw prediction tensors into the ``TimeSeries`` datatype.

        Rows are labelled with the series they were forecast for and the time
        index they fall on, so that the result can be joined back to the data
        the forecast was made for.
        """
        y = out["prediction"]
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        n_windows, prediction_length = y.shape[0], y.shape[1]
        y = y.reshape(n_windows * prediction_length, -1)

        # the last dimension holds the targets in point prediction mode only;
        # for quantiles it holds the quantiles, so the target names do not fit
        metadata = datamodule.time_series_metadata
        if y.shape[-1] != len(metadata["cols"]["y"]):
            metadata = None

        groups, t = self._prediction_index(datamodule, n_windows, prediction_length)
        return TimeSeries_datatype.from_tensors(
            {"y": y, "t": t}, metadata=metadata, groups=groups
        )

    @staticmethod
    def _prediction_index(datamodule, n_windows: int, prediction_length: int):
        """Series and time index of every forecast row.

        Taken from the windows the data module built for prediction, each of
        which is ``(series_idx, start_idx, encoder_length, prediction_length)``.
        If those are not available, or do not line up with the predictions,
        each window becomes its own series with a horizon-local time index -
        wrong provenance would be worse than none.
        """
        windows = getattr(datamodule, "predict_windows", None)

        if windows is None or len(windows) != n_windows:
            groups = np.repeat(np.arange(n_windows), prediction_length)
            t = np.tile(np.arange(prediction_length), n_windows)
            return groups, t

        groups = np.repeat([w[0] for w in windows], prediction_length)
        t = np.concatenate(
            [np.arange(w[1] + w[2], w[1] + w[2] + prediction_length) for w in windows]
        )
        return groups, t
