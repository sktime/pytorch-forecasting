import torch

from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Baseline_v2(TslibBaseModel):
    @classmethod
    def _pkg(cls):
        """Get package class."""
        from pytorch_forecasting.models.baseline._baseline_pkg_v2 import Baseline_pkg_v2

        return Baseline_pkg_v2

    def __init__(self, loss=None, logging_metrics=None, metadata=None, **kwargs):
        if loss is None:
            loss = MAE()
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            metadata=metadata,
        )

        # self.automatic_optimization = False

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        pass  # no network to initialize for baseline model

    def configure_optimizers(self):
        return None  # no optimizer needed for baseline model

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "history_target" not in x or x["history_target"].size(1) == 0:
            raise ValueError(
                "Input dictionary must contain non-empty 'history_target'."
            )

        history_target = x["history_target"]
        last_value = history_target[:, -1:, :]
        prediction = last_value.repeat(1, self.prediction_length, 1)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}

    def training_step(self, batch, batch_idx):
        super().training_step(batch, batch_idx)
        return None
