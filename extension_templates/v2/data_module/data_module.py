"""Extension Template for Data Module (D2 Layer)
Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
    - if the name has more than one word (like Encoder-Decoder Data Module), the name of
    the file should be created by separating these words by a underscore (_).
    For eg, for Encoder-Decoder Data Module, the name of the file would be
    encoder_decoder/_encoder_decoder_data_module.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement (abstract on BaseTimeSeriesDataModule):
    _prepare_metadata
        Derive model-init metadata (shapes, feature counts, window lengths, etc.)
        from the D1 TimeSeries and datamodule hyper-parameters.
    _context_length
        Return encoder/context window size.
    _prediction_length
        Return decoder/prediction window size.
    _create_windows(indices)
        Build sliding-window index tuples
        ``(series_idx, start_idx, context_length, prediction_length)`` for the
        given series indices. Skip series shorter than context + prediction.
    _build_dataset(windows)
        Return a private processed ``torch.utils.data.Dataset`` for the windows.
    _ensure_split()
        Split series indices into train, validation, and test tensors using
        ``self.train_val_test_split`` once and cache them.
        Cache them in ``self._train_indices``, ``self._val_indices``, and
        ``self._test_indices``.
    collate_fn(batch)
        Static method that stacks dataset samples into the ``(x, y)`` batch
        layout expected by your model's ``forward`` pass.

Provided by BaseTimeSeriesDataModule (override only when needed):
    _preprocess_data(series_idx)
        Load one series, tensorize ``y``/``x``, apply cutoff mask.
        Used by the processed dataset.
    metadata (property)
        Lazy cache around ``_prepare_metadata()``.
    setup(stage)
        Calls ``_ensure_split()``, builds windows/datasets per stage
        (``fit``, ``test``, ``predict``).
    train_dataloader / val_dataloader / test_dataloader / predict_dataloader
        Standard Lightning dataloaders wired to ``collate_fn``.
"""

# todo: write an informative docstring for the file or module, remove the above

import torch
from torch.utils.data import Dataset

from pytorch_forecasting.data.data_module import BaseTimeSeriesDataModule

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file


class MyDataModule(BaseTimeSeriesDataModule):
    """Custom DataModule.
    todo: write docstring.

    todo: describe your custom DataModule here

    Parameters
    ----------
    parama : anytype
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on
    """

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama=None, paramb="default", paramc=None):
        # collect all the params passed to __init__() below
        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc
        # leave this as is — pass time_series_dataset (D1 TimeSeries) and
        # other base kwargs here
        super().__init__()
        # create any other required params after this

    # implement this is mandatory
    def _prepare_metadata(self) -> dict:
        """Prepare metadata for model initialisation.

        Returns
        -------
        dict
            Dictionary containing the params required to initialise the model.
            # todo: add all the keys that the dict has
        """
        # collect all the keys that are required for the model initialisation and
        # can be derived in any way from the dataset
        #
        # This could be info that the user might have already provided while
        # intializing the `TimeSeries` dataset class or this data module (from __init__)
        # For eg, while initializing the `TimeSeries` dataset class, the user would've
        # already provided what are static variables in the data.
        # You might not want to add the exact keys that `TimeSeries` provide through its
        # metadata into this method, rather parse them to get new information like
        # combining the information that col1 and col3 are static, but only col1 is
        # categorical to create a new key called static_categorical which has only col1
        #
        # Another way could be that some information can be derived from the input of
        # data module - you might need to perform any basic operation on the data to
        # derive this info.

    def _context_length(self) -> int:
        """Return encoder/context window length."""
        # todo: return the context window length

    def _prediction_length(self) -> int:
        """Return decoder/prediction window length."""
        # todo: return the prediction window length

    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Build sliding-window index tuples for the given series indices."""
        # todo: return the sliding-window index tuples

    def _build_dataset(self, windows: list[tuple[int, int, int, int]]) -> Dataset:
        """Return a processed Dataset for DataLoader consumption."""
        # todo: return your private _MyDataset(self, windows)

    def _ensure_split(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split series indices into train, val, and test sets based on the
        train_val_test_split ratio once and cache them.

        Sets
        -------
        sets the following attributes:

        - ``_train_indices``
        - ``_val_indices``
        - ``_test_indices``

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The train, validation, and test indices.
        """
        # todo: implement using self.train_val_test_split

    @staticmethod
    def collate_fn(batch):
        """Stack samples from the processed dataset into a model-ready batch."""
        # todo: implement

    # Optional overrides:
    # - _preprocess_data(series_idx) — only if base tensorization is insufficient
    # - train_shuffle (property) — return False to disable training shuffle
