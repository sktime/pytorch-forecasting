"""Template for a custom data module in the v2 time-series pipeline."""

from typing import Any

import torch

from pytorch_forecasting.data.data_module import BaseTimeSeriesDataModule

from ._dataset import _MyDataModuleDataset


class MyDataModule(BaseTimeSeriesDataModule):
    """Custom data module implementing the v2 time-series data-module hooks.

    The base class owns data attachment, split setup, and dataloader creation.
    Implement only the hooks needed to prepare metadata and model-ready batches.
    """

    def _prepare_metadata(self) -> Any:
        """Derive model metadata from the attached time series and config."""
        raise NotImplementedError("Derive model metadata for this data module.")

    def _preprocess_data(self, series_idx: int | torch.Tensor) -> dict[str, Any]:
        """Preprocess and cache one series for the windowed dataset."""
        raise NotImplementedError("Preprocess one series.")

    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Create windows as (series, start, context length, prediction length)."""
        raise NotImplementedError("Create windows for the requested series indices.")

    def _build_dataset(self, indices: torch.Tensor) -> _MyDataModuleDataset:
        """Build a dataset from series indices selected by the base module."""
        return _MyDataModuleDataset(self, self._create_windows(indices))

    @staticmethod
    def collate_fn(
        batch: list[
            tuple[
                dict[str, torch.Tensor],
                torch.Tensor | list[torch.Tensor],
            ]
        ],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        """Collate samples into the ``(x, y)`` batch expected by the model."""
        raise NotImplementedError("Collate samples into model inputs and targets.")
