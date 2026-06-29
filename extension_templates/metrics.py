"""Extension Template for metrics
Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new metric:
- make a copy of the template in a suitable location, give it a descriptive name.
    - if the name has more than one word (like abc foo metric), the name of
    the file should be created by separating these words by a underscore (_).
    For eg, for abc-foo metric, the name of the file would be
    _abc_foo_metric.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Base Classes to choose from:
Pytorch-forecasting provides different Base classes that can be used based upon what
kind of metric is being implemented. Some supported Base Classes to choose from:
- MultiHorizonMetric - If you want to implement point prediction metric or any genric
    metric
- DistributionLoss/MultivariateDistributionLoss - For DistributionLoss

Mandatory methods to implement:
    loss - to calculate loss

Optional Methods to implement:
    __init__ - if you need to pass special args to initialize the metric
    rescale_parameters - rescale the parameter values so that the loss can be computed.
        Often implemented for DistributionLoss metrics
    map_x_to_distribution - map x to the distribution
        Often implemented for DistributionLoss metrics
    to_prediction - Convert network prediction into prediction as per the loss.
        Implemented if you need some special preprocessing to return the prediction
        Eg, for point prediction, we take argmax - y_pred.argmax(dim=-1)
    to_quantile - Convert network prediction into a quantile prediction.
        Implemented if you need some special preprocessing to return the quantile pred.
"""

# todo: write an informative docstring for the file or module, remove the above
import torch
from torch import distributions

from pytorch_forecasting.metrics import MultiHorizonMetric

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file


class MyMetric(MultiHorizonMetric):
    """Custom Metric.
    todo: write docstring.

    todo: describe your custom metric here

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

    # OPTIONAL method
    # create __init__() if you need specific args to be passed to initilialize the class
    def __init__(self, parama, paramb, paramc):
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

    # implement this is mandatory
    def loss(self, paramx):
        """loss computation.

        todo: write a descriptive docstring

        Parameters
        ----------
        paramx : anytype
          descriptive explanation of paramx
        and so on
        """
        # implement the loss function

    # todo implement the other optional methods if needed otherwise delete the following
    # lines
    # Other optional methods
    # def rescale_parameters(self, param):
    #     """rescale the parameter values so that the loss can be computed.
    #     todo write docstring
    #     """
    #     # implement the method
    #
    # def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
    #     """map x to distribution
    #     todo write docstring
    #     """
    #     # implement the method
    #
    # def to_prediction(self, x: torch.Tensor) -> torch.Tensor:
    #     """convert x to prediction
    #     todo write docstring
    #     """
    #     # implement the method
    #
    # def to_quantile(self, x: torch.Tensor) -> torch.Tensor:
    #     """convert x to quantile
    #     todo write docstring
    #     """
    #     #implement the method
