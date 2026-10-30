"""Extension template for the private dataset used by a custom data module.

Purpose of this implementation template:
    quick implementation of new datasets following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new dataset:
- copy this file together with data_module.py and rename the dataset class
- work through all the "todo" comments below
- implement sample construction to match the data module's collate_fn
- update constructor arguments, item-format descriptions, and docstrings
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods:
    __len__ - return the number of available forecast windows.
    __getitem__ - construct one model input/target pair from a window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .data_module import MyDataModule


# TODO: Rename the class and update the matching data-module import and factory.
class _MyDataModuleDataset(
    Dataset[tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]]
):
    """Dataset returning one model input/target pair for each forecast window.

    TODO: Describe the processed series representation, input keys, targets, and
    windowing assumptions used by the implemented model.

    Parameters
    ----------
    data_module : MyDataModule
        Parent data module providing configuration, metadata, and preprocessing.
    windows : list[tuple[int, int, int, int]]
        Forecast windows as (series index, start index, context length,
        prediction length), produced by the data module's _create_windows.
    """

    # TODO: Add and document additional configuration only if this dataset needs it.
    def __init__(
        self,
        data_module: MyDataModule,
        windows: list[tuple[int, int, int, int]],
    ) -> None:
        self.data_module = data_module
        self.windows = windows

    def __len__(self) -> int:
        """Return the number of forecast windows available to the dataloader."""
        return len(self.windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        """Return one ``(x, y)`` pair in the format expected by ``collate_fn``.

        Parameters
        ----------
        idx : int
            Index into the forecast-window descriptors.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor or list[torch.Tensor]]
            Input features and the corresponding target values. TODO: Document
            each feature key and tensor shape, including multiple targets.
        """
        # TODO: select this window, preprocess its series, and create x and y.
        # Use the parent module's preprocessing and keep sample construction
        # consistent with the module's collate_fn and the model's forward method.
        raise NotImplementedError("Build one model input/target pair.")

    # TODO: Add documented helpers for sample construction if necessary.
