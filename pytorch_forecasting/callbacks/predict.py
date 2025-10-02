from typing import Any, Optional
from warnings import warn

from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import torch

from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class PredictCallback(BasePredictionWriter):
    """
    Callback to capture predictions and related information internally.

    This callback is used by `BaseModel.predict()` to process raw model outputs
    into the desired format (`prediction`, `quantiles`, or `raw`) and collect
    any additional requested info (`x`, `y`, `index`, etc.). The results are
    collated and stored in memory, accessible via the `.result` property. It does
    not write to disk.

    Parameters
    ----------
    mode : str
        The prediction mode ("prediction", "quantiles", or "raw").
    return_info : list[str], optional
        Additional information to return.
    **kwargs :
        Additional keyword arguments for `to_prediction` or `to_quantiles`.
    """

    def __init__(
        self,
        mode: str = "prediction",
        return_info: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(write_interval="epoch")
        self.mode = mode
        self.return_info = return_info or []
        self.kwargs = kwargs
        self._reset_data()

    def _reset_data(self):
        """Clear collected data for a new prediction run."""
        self.predictions = []
        self.info = {key: [] for key in self.return_info}
        self._result = None

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModel,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Process and store predictions for a single batch."""
        x, y = batch

        if self.mode == "raw":
            processed_output = outputs
        elif self.mode == "prediction":
            processed_output = pl_module.to_prediction(outputs, **self.kwargs)
        elif self.mode == "quantiles":
            processed_output = pl_module.to_quantiles(outputs, **self.kwargs)
        else:
            raise ValueError(f"Invalid prediction mode: {self.mode}")

        self.predictions.append(processed_output)

        for key in self.return_info:
            if key == "x":
                self.info[key].append(x)
            elif key == "y":
                self.info[key].append(y[0])
            elif key == "index":
                self.info[key].append(y[1])
            elif key == "decoder_lengths":
                self.info[key].append(x["decoder_lengths"])
            else:
                warn(f"Unknown return_info key: {key}")

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: "BaseModel"):
        """Collate all batch results into final tensors."""
        if self.mode == "raw" and isinstance(self.predictions[0], dict):
            keys = self.predictions[0].keys()
            collated_preds = {
                key: torch.cat([p[key] for p in self.predictions]) for key in keys
            }
        else:
            collated_preds = {"prediction": torch.cat(self.predictions)}

        final_result = collated_preds

        for key, data_list in self.info.items():
            if isinstance(data_list[0], dict):
                collated_info = {
                    k: torch.cat([d[k] for d in data_list]) for k in data_list[0].keys()
                }
            else:
                collated_info = torch.cat(data_list)
            final_result[key] = collated_info

        self._result = final_result
        self._reset_data()

    @property
    def result(self) -> dict[str, torch.Tensor]:
        if self._result is None:
            raise RuntimeError("Prediction results are not yet available.")
        return self._result
