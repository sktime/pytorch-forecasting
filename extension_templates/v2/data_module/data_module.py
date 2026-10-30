"""Extension template for a custom data module (D2 layer).

Purpose of this implementation template:
    quick implementation of new data modules following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new data module:
- make a copy together with _dataset.py and give the module a descriptive name
- work through all the "todo" comments below
- implement the data-format hooks required by BaseTimeSeriesDataModule
- update the metadata, batch-format, parameter, and preprocessing descriptions
- select this class in the corresponding forecaster's get_datamodule_cls method
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory hooks to implement:
    _prepare_metadata - derive the metadata required by the model constructor.
    _preprocess_data - preprocess one series for dataset consumption.
    _create_windows - define forecast windows over selected series.
    _build_dataset - construct the private dataset for those windows.
    collate_fn - collate dataset samples into an (x, y) batch.

The base owns metadata access, with_data, setup, and dataloader creation. Inherit
those methods instead of reimplementing their data-attachment and split logic.
"""

from typing import Any

import torch

from pytorch_forecasting.data.data_module import BaseTimeSeriesDataModule

from ._dataset import _MyDataModuleDataset


# TODO: Rename the class and describe the implemented data format.
class MyDataModule(BaseTimeSeriesDataModule):
    """Custom data module implementing the v2 time-series data-module hooks.

    The base class owns data attachment, split setup, and dataloader creation.
    TODO: Describe the supported series, preprocessing, model metadata, and batch
    keys. Complete every required hook below before using the adapted class.

    Parameters
    ----------
    time_series : TimeSeries, optional
        Input data. Leave unset when configuring a forecaster; its fit and
        predict methods attach data through with_data.
    target_normalizer : normalizer, str, list, tuple or None, default="auto"
        Target normalization setting interpreted by the base data module.
    batch_size : int, default=32
        Batch size used by the inherited dataloaders.
    num_workers : int, default=0
        Number of dataloader workers.
    train_val_test_split : tuple[float, float, float], default=(0.7, 0.15, 0.15)
        Fractions used by the base for train, validation, and test splits.
    add_relative_time_idx : bool, default=False
        Whether to include the relative-time-index feature.
    """

    # The inherited constructor exposes the options above and records them for
    # with_data. Only add an override if the module needs extra configuration.
    # TODO: If overriding, expose supported base options explicitly, store new
    # options as same-named attributes before super(), and forward the base options.
    # Hiding base options in **kwargs would prevent their capture for cloning.

    # Mandatory: derive metadata consumed by the model constructor.
    def _prepare_metadata(self) -> Any:
        """Derive model metadata from the attached time series and config.

        TODO: Describe each metadata field and how it is derived. Combine the
        attached TimeSeries metadata with this module's configuration to provide
        the dimensions, feature groups, and other information the model needs.
        For example, derive static categorical features from the supplied static
        and categorical column information instead of hardcoding their counts.

        Returns
        -------
        Any
            TODO: Specify the concrete metadata type and its fields, matching
            the model's metadata argument. The base property caches this result.
        """
        # TODO: Parse the data schema and derive the model-specific metadata.
        raise NotImplementedError("Derive model metadata for this data module.")

    # Mandatory: define preprocessing for a single series.
    def _preprocess_data(self, series_idx: int | torch.Tensor) -> dict[str, Any]:
        """Preprocess and cache one series for the windowed dataset.

        Parameters
        ----------
        series_idx : int or torch.Tensor
            Index of the time series to preprocess.

        Returns
        -------
        dict[str, Any]
            TODO: Document the processed features, targets, static values, and
            other entries read by the private dataset.

        Preprocessing steps
        -------------------
        TODO: Describe normalization, feature transformation, and caching steps.
        """
        # TODO: Prepare one series in the form consumed by _MyDataModuleDataset.
        raise NotImplementedError("Preprocess one series.")

    # Mandatory: enumerate forecast windows for the selected series.
    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Create windows as (series, start, context length, prediction length).

        Parameters
        ----------
        indices : torch.Tensor
            Series indices selected by the base module for the current split.

        Returns
        -------
        list[tuple[int, int, int, int]]
            Window descriptors: series index, start index, context length, and
            prediction length. TODO: Explain split boundaries and window rules.
        """
        # TODO: Select valid windows without leaking observations between splits.
        raise NotImplementedError("Create windows for the requested series indices.")

    # Mandatory: connect the windows to the private dataset implementation.
    def _build_dataset(self, indices: torch.Tensor) -> _MyDataModuleDataset:
        """Build a dataset from series indices selected by the base module.

        TODO: Update the dataset class name and constructor if its inputs change.
        Keep window creation consistent with the selected base-module split.
        """
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
        """Collate samples into the ``(x, y)`` batch expected by the model.

        Parameters
        ----------
        batch : list of tuple
            Samples returned by the private dataset's __getitem__ method.

        Returns
        -------
        tuple
            Batched input dictionary and target tensor, or list of target tensors.
            TODO: Document input keys, tensor shapes, lengths, and padding rules.
        """
        # TODO: Stack or pad samples into the format required by forward and loss.
        raise NotImplementedError("Collate samples into model inputs and targets.")

    # TODO: Add documented helpers for any additional data-format-specific logic.
