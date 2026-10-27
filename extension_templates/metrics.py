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
        Choose the conversion for the implemented loss; classification scores
        and numeric forecasts need different conversions.
    to_quantiles - Convert network prediction into quantile predictions.
        Implemented if you need some special preprocessing to return the quantile pred.
"""

from typing import Any

import torch

from pytorch_forecasting.metrics import MultiHorizonMetric

# TODO: Add imports needed by the implemented metric.
# Import optional dependencies inside methods rather than at module level.


# TODO: Rename the class and replace the placeholder documentation below.
class MyMetric(MultiHorizonMetric):
    """Custom Metric.
    todo: write docstring.

    todo: describe your custom metric here

    Parameters
    ----------
    parama : Any, optional
        TODO: Describe the metric-specific option and its default.
    paramb : str, default="default"
        TODO: Describe this option and its valid values.
    paramc : Any, optional
        TODO: Describe any additional component or configuration.
    **kwargs : Any
        Supported base metric options, such as reduction.
    """

    # Optional: add parameters specific to this metric.
    # TODO: Replace the example parameters and update their descriptions above.
    def __init__(
        self,
        parama: Any = None,
        paramb: str = "default",
        paramc: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Collect constructor parameters as attributes. Do not overwrite or
        # mutate them to resolve defaults; use separate private attributes instead.
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

    # Mandatory: compute per-sample, per-horizon losses before base reduction.
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

        TODO: Describe the implemented formula and any prediction/target shape
        requirements specific to this metric.
        """
        # TODO: implement loss using the existing multi-horizon tensor contract.
        raise NotImplementedError("Implement the metric loss.")

    # TODO: Implement optional methods only when needed; otherwise remove the
    # corresponding example comments from the adapted metric.
    # Implement these only when the metric needs custom prediction conversion.
    #
    # def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
    #     """Convert network output to a point prediction."""
    #     # TODO: Apply the conversion required by this metric.
    #
    # def to_quantiles(
    #     self, y_pred: torch.Tensor, quantiles: list[float] | None = None
    # ) -> torch.Tensor:
    #     """Convert network output to quantile predictions."""
    #     # TODO: Produce the requested quantiles from this metric's predictions.
    #
    # DistributionLoss subclasses can additionally implement
    # map_x_to_distribution and rescale_parameters as required by their base class.
    # TODO: When adapting a distribution loss, copy the corresponding base-class
    # signatures and document the distribution and parameter rescaling used.
