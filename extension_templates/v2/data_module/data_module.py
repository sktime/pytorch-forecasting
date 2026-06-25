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
    _encoder_decoder_data_module.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement:
    _prepare_metadata - method to create metadata
    metadata - property to access the metadata
    _preprocess_data - method to preprocess data
    setup - method to setup the ML pipeline
"""

# todo: write an informative docstring for the file or module, remove the above
from typing import Any

from lightning.pytorch import LightningDataModule
import torch

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file


class MyDataModule(LightningDataModule):
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
        # leave this as is
        super().__init__()
        # create anyother required params after this
        # must have this arg in self
        self._metadata = None

    # implement this is mandatory
    def _prepare_metadata(self):
        """Prepare metadata for model initialisation.

        Returns
        -------
        dict
            dictionary containing the params required to initialise the model.
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

    # implement this is mandatory
    @property
    def metadata(self):
        """Compute metadata for model initialization.

        This property returns a dictionary containing the shapes and key information
        related to the time series model.
        # todo add all the keys that the metadata has"""
        # you can keep this method as it is. It just takes _prepare_metadata() to create
        # this property
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    # implement this is mandatory
    def _preprocess_data(self, series_idx: torch.Tensor) -> list[dict[str, Any]]:
        """Preprocess the data before feeding it into _ProcessedEncoderDecoderDataset.

        Preprocessing steps
        --------------------
        # todo: document all the processing steps
        """
        # Add the preprocessing of data here that would be then passed to a private
        # _Mydataset class, see datamodule/_dataset.py for more info.

    # implement this is mandatory
    def setup(self, stage: str) -> None:
        """Setup the DataModule.
        todo: implement the DataModule.setup() method. Add complete docstring."""
        # implement the setup method and handle different stages of the ML pipeline
        # (train, test, predict, validation etc) accordingly.

    # If needed create collate_fn, dataloader methods and other required helping methods
