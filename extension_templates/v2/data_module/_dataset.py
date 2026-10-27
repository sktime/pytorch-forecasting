"""Template for the private windowed dataset used by a custom data module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .data_module import MyDataModule


class _MyDataModuleDataset(
    Dataset[tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]]
):
    """Dataset returning one model input/target pair for each forecast window."""

    def __init__(
        self,
        data_module: MyDataModule,
        windows: list[tuple[int, int, int, int]],
    ) -> None:
        self.data_module = data_module
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        """Return one ``(x, y)`` pair in the format expected by ``collate_fn``."""
        # TODO: select this window, preprocess its series, and create x and y.
        raise NotImplementedError("Build one model input/target pair.")
