"""Extension template for metrics.

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

Base classes to choose from:
- MultiHorizonMetric for point-prediction or other multi-horizon metrics.
- DistributionLoss or MultivariateDistributionLoss for distribution losses.

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
    to_quantiles - Convert network prediction into quantile predictions.
        Implemented if you need some special preprocessing to return the quantile pred.
"""

from typing import Any

import torch

from pytorch_forecasting.metrics import MultiHorizonMetric


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

    # Optional: add parameters specific to this metric.
    def __init__(
        self,
        parama: Any = None,
        paramb: str = "default",
        paramc: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute unreduced loss values for a batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Prediction tensor supplied by ``BaseModel``.
        target : torch.Tensor
            Target values corresponding to the model output.

        Returns
        -------
        torch.Tensor
            Loss values before the base metric applies masking and reduction.
        """
        # TODO: implement loss using the existing multi-horizon tensor contract.
        raise NotImplementedError("Implement the metric loss.")

    # Implement these only when the metric needs custom prediction conversion.
    #
    # def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
    #     """Convert network output to a point prediction."""
    #
    # def to_quantiles(
    #     self, y_pred: torch.Tensor, quantiles: list[float] | None = None
    # ) -> torch.Tensor:
    #     """Convert network output to quantile predictions."""
    #
    # DistributionLoss subclasses can additionally implement
    # map_x_to_distribution and rescale_parameters as required by their base class.
