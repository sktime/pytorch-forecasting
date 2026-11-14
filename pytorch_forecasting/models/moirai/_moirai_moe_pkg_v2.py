"""Package container for zero-shot Moirai-MoE inference."""

from lightning import LightningDataModule

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.models.moirai._moirai_moe_v2 import MoiraiMoE


class MoiraiMoE_pkg_v2(Base_pkg):
    """Load a pretrained Moirai-MoE model on the first call to ``predict``.

    Pass a ``TimeSeries`` with ``datamodule_cfg``, or a ``TslibDataModule``,
    for the first prediction so the model can obtain its metadata. Subsequent
    predictions also accept a DataLoader with the same context and target layout.
    ``model_cfg`` accepts the parameters of :class:`MoiraiMoE`.
    """

    _tags = {
        "info:name": "MoiraiMoE",
        "info:compute": 4,
        "authors": ["andersendsa"],
        "python_dependencies": ["uni2ts>=2.0,<3"],
        "python_version": "<3.13",
        "info:y_type": ["numeric"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
        # These generic tests require gradient training and training checkpoints.
        "tests:skip_by_name": [
            "test_integration",
            "test_checkpointing",
            "test_predict_modes",
        ],
    }

    @classmethod
    def get_cls(cls):
        return MoiraiMoE

    @classmethod
    def get_datamodule_cls(cls):
        return TslibDataModule

    def fit(self, *args, **kwargs):
        """Reject training; use ``predict`` directly for zero-shot forecasts."""
        raise NotImplementedError(
            "MoiraiMoE supports zero-shot prediction only; call predict directly."
        )

    def predict(self, data, output_dir=None, **kwargs):
        """Initialize from dataset metadata and use the standard prediction pipeline."""
        if isinstance(data, TimeSeries):
            data = self._build_datamodule(data)
        if self.model is None:
            if not isinstance(data, LightningDataModule):
                raise ValueError(
                    "The first prediction requires a TimeSeries or TslibDataModule "
                    "to obtain model metadata."
                )
            self.datamodule = data
            self.model = self.get_cls()(metadata=data.metadata, **self.model_cfg)
        elif isinstance(data, LightningDataModule):
            metadata = data.metadata
            if (
                metadata["context_length"] != self.model.context_length
                or metadata["prediction_length"] != self.model.prediction_length
                or metadata["n_features"]["target"] != self.model.target_dim
            ):
                raise ValueError(
                    "Prediction data must match the initialized context length, "
                    "prediction length, and target dimension. Create a new forecaster "
                    "for different metadata."
                )
        kwargs.setdefault("trainer_kwargs", self.trainer_cfg.copy())
        return super().predict(data, output_dir=output_dir, **kwargs)
