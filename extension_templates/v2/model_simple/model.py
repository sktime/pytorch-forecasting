"""Extension template for models.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
    - if the name has more than one word (like Temporal Fusion Transformer), the name of
    the file should be created by separating these words by a underscore (_).
    For eg, for Temporal Fusion Transformer model, the name of the file would be
    temporal_fusion_transformer.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseModel's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by
    pytorch-forecasting.utils._estimator_checks.check_estimator
- once complete: use as a local library, or contribute to pytorch-forecasting via PR
- IMPORTANT: if you have some custom layers that are used by the model, you should add
    that to `pytorch-forecasting.layers` module and then import that layer in this file.

Mandatory methods to implement:
    forward - the forward pass of the model
    _pkg - method to access the package class of the model

Testing - required for pytorch-forecasting test framework and check_estimator usage:
    Use `model_pkg` class for this. See model_pkg.py in the same folder for more info
"""

# todo: write an informative docstring for the file or module, remove the above

import torch

from pytorch_forecasting.models.base._base_model_v2 import BaseModel

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file
# donot import the model class at the module level, it should
# be imported within the respective method to access that classes (namely _pkg).


# todo: change class name and write docstring
class MyModel(BaseModel):
    """Custom Model.
    todo: write docstring.

    todo: describe your custom forecaster here

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
    # All the params of __init__() should ideally have a "good" default value
    def __init__(self, parama=None, paramb="default", paramc=None):
        # save the hparams, you can ignore some of them
        # for example
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "optimizer"])
        # add any params to super.__init__() that you want to pass
        # to the parent BaseModel
        # example:
        # super().__init__(
        #             loss=loss,
        #             logging_metrics=logging_metrics,
        #             optimizer=optimizer,
        #             optimizer_params=optimizer_params,
        #             lr_scheduler=lr_scheduler,
        #             lr_scheduler_params=lr_scheduler_params,
        # )
        # Look at the __init__() of BaseModel to see which params we need to pass to
        # super().__init__()
        super().__init__(loss=parama)  # or anyother param

        # collect all the params passed to __init__() below
        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc

        # create anyother required params (like self.linear_layer etc) after this

    # implement this is mandatory
    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        # import the package class from the model package file
        from pytorch_forecasting.models.my_model.model_pkg import MyModel_pkg

        return MyModel_pkg

    # implement this is mandatory
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # todo: write a clear docstring in numpydoc style
        """
        Forward pass of the model.

        Parameters
        ---------
        x : dict[str, torch.Tensor]
            input data generated from the data loaders of data modules
        """
        # todo: implement the forward loop

    # implement any helping methods for the class
