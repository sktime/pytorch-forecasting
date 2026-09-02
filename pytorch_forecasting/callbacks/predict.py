from typing import Any, Optional
from warnings import warn

from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import torch

from pytorch_forecasting.utils import detach, move_to_device

_PRIVATE_WINDOW_KEY = "__window_idx"


def _concat_batch_values(values: list[Any]) -> Any:
    """Concatenate values collected from prediction batches."""
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    if isinstance(first, pd.DataFrame):
        return pd.concat(values, ignore_index=True)
    if isinstance(first, dict):
        return {
            key: _concat_batch_values([value[key] for value in values]) for key in first
        }
    if isinstance(first, list):
        return [
            _concat_batch_values([value[idx] for value in values])
            for idx in range(len(first))
        ]
    if isinstance(first, tuple):
        return tuple(
            _concat_batch_values([value[idx] for value in values])
            for idx in range(len(first))
        )
    raise TypeError(f"Unsupported prediction info type: {type(first).__name__}")


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
        elif self.mode == "prediction":
            processed_output = pl_module.to_prediction(outputs, **self.mode_kwargs)
        elif self.mode == "quantiles":
            processed_output = pl_module.to_quantiles(outputs, **self.mode_kwargs)
        else:
            raise ValueError(f"Invalid prediction mode: {self.mode}")

        self.predictions.append(move_to_device(detach(processed_output), "cpu"))

        needs_x = any(k in ("x", "index", "decoder_lengths") for k in self.return_info)
        x_cpu = move_to_device(detach(x), "cpu") if needs_x else None

        for key in self.return_info:
            if key == "x":
                public_x = {k: v for k, v in x_cpu.items() if k != _PRIVATE_WINDOW_KEY}
                self.info[key].append(public_x)
            elif key == "y":
                self.info[key].append(move_to_device(detach(y), "cpu"))
            elif key == "index":
                dataset = getattr(trainer.predict_dataloaders, "dataset", None)
                if dataset is None or not hasattr(dataset, "x_to_index"):
                    raise TypeError(
                        "return_info=['index'] requires the prediction dataset to "
                        "implement x_to_index(x)."
                    )
                self.info[key].append(dataset.x_to_index(x_cpu))
            elif key == "decoder_lengths":
                if "decoder_lengths" in x_cpu:
                    lengths = x_cpu["decoder_lengths"]
                elif "future_length" in x_cpu:
                    lengths = x_cpu["future_length"]
                else:
                    raise KeyError(
                        "Prediction batch does not provide 'decoder_lengths' or "
                        "'future_length'."
                    )
                self.info[key].append(lengths)
            else:
                warn(f"Unknown return_info key: {key}")

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
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
            final_result[key] = _concat_batch_values(data_list)

        self._result = final_result
        self._reset_data(result=False)

    @property
    def result(self) -> dict[str, torch.Tensor]:
        if self._result is None:
            raise RuntimeError("Prediction results are not yet available.")
        return self._result
