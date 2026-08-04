from typing import Any, Optional
from warnings import warn

from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
import torch

from pytorch_forecasting.utils import detach, move_to_device


class PredictCallback(BasePredictionWriter):
    """
    Callback to capture predictions and related information internally.

    This callback is used by ``BaseModel.predict()`` to process raw model outputs
    into the desired format (``prediction``, ``quantiles``, or ``raw``) and collect
    any additional requested info (``x``, ``y``, ``index``, etc.). The results are
    collated and stored in memory, accessible via the ``.result`` property.

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
        return_info: list[str] | None = None,
        mode_kwargs: dict[str, Any] = None,
    ):
        super().__init__(write_interval="epoch")
        self.mode = mode
        self.return_info = return_info or []
        self.mode_kwargs = mode_kwargs or {}
        self._reset_data()

    def _reset_data(self, result: bool = True):
        """Clear collected data for a new prediction run."""
        self.predictions = []
        self.info = {key: [] for key in self.return_info}
        if result:
            self._result = None

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Process and store predictions for a single batch."""
        x, y = batch

        if self.mode == "raw":
            processed_output = outputs
            if not isinstance(processed_output["prediction"], (list, tuple)):
                processed_output["prediction"] = [processed_output["prediction"]]
        elif self.mode == "prediction":
            processed_output = pl_module.to_prediction(outputs, **self.mode_kwargs)
        elif self.mode == "quantiles":
            processed_output = pl_module.to_quantiles(outputs, **self.mode_kwargs)
        else:
            raise ValueError(f"Invalid prediction mode: {self.mode}")

        self.predictions.append(move_to_device(detach(processed_output), "cpu"))

        # Only pay the detach+copy cost if x or decoder_lengths are actually requested
        needs_x = any(k in ("x", "decoder_lengths") for k in self.return_info)
        x_cpu = move_to_device(detach(x), "cpu") if needs_x else None

        for key in self.return_info:
            if key == "x":
                self.info[key].append(x_cpu)
            elif key == "y":
                y_cpu = move_to_device(detach(y[0]), "cpu")
                self.info[key].append(y_cpu)
            elif key == "index":
                index_cpu = move_to_device(detach(y[1]), "cpu")
                self.info[key].append(index_cpu)
            elif key == "decoder_lengths":
                self.info[key].append(x_cpu["decoder_lengths"])
            else:
                warn(f"Unknown return_info key: {key}")

    @staticmethod
    def _collate(items: list[Any]) -> Any:
        """Recursively concatenate a list of per-batch tensors/dicts/lists into one."""
        first = items[0]
        if isinstance(first, dict):
            return {
                key: PredictCallback._collate([item[key] for item in items])
                for key in first
            }
        if isinstance(first, (list, tuple)):
            n = len(first)
            return [
                PredictCallback._collate([item[i] for item in items]) for i in range(n)
            ]
        return torch.cat(items)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Collate all batch results into final tensors."""
        is_raw_dict = self.mode == "raw" and isinstance(self.predictions[0], dict)
        collated_preds = self._collate(self.predictions)

        final_result = collated_preds if is_raw_dict else {"prediction": collated_preds}

        for key, data_list in self.info.items():
            final_result[key] = self._collate(data_list)

        self._result = final_result
        self._reset_data(result=False)

    @property
    def result(self) -> dict[str, torch.Tensor]:
        if self._result is None:
            raise RuntimeError("Prediction results are not yet available.")
        return self._result
