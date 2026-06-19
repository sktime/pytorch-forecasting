"""Extension Template for the private Dataset class
Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement:
    __getitem__
"""

# todo: write an informative docstring for the file or module, remove the above
from torch.utils.data import Dataset

from extension_templates.v2.data_module.data_module import MyDataModule

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file


class _myDataModuleDataset(Dataset):
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
    def __init__(self, data_module: "MyDataModule", paramb, paramc):
        # collect all the params passed to __init__() below
        # todo: write any hyper-parameters and components to self
        self.data_module = (
            data_module  # collect the datamodule in which we are going to
        )
        # use this dataset
        self.paramb = paramb
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc

    # implement this is mandatory
    def __getitem__(self, idx):
        """Get the processed dataset item at the given index.
        todo: write docstring.
        """
        # write the __getitem__ of the dataset class in the way you expect the
        # collate_fn to be, and you want to dataloader to be loaded.
