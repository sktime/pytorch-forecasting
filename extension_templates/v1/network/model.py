"""Extension template for v1 neural network models.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseModel's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR
- IMPORTANT: if you have some custom layers that are used by the model, you should add
    that to a ``layers/`` subfolder in your model directory and import from there.

Mandatory methods to implement:
    __init__ - constructor with model hyperparameters
    forward - the forward pass of the model
    _pkg - method to access the package class of the model
    from_dataset - factory method to construct model from a TimeSeriesDataSet

Optional methods (delete if not needed):
    to_prediction - custom post-processing of point predictions
    to_quantiles - custom quantile extraction for probabilistic outputs

Testing - required for pytorch-forecasting test framework:
    Use the ``_pkg`` class for this. See _model_pkg.py for more info.
"""

# todo: write an informative docstring for the file or module, remove the above

import torch

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

# Choose the appropriate base class:
# - Use ``BaseModel`` if the model does NOT use/support covariates or autoregressive
#   features (e.g., N-BEATS).
# - Use ``BaseModelWithCovariates`` if the model supports static and/or time-varying
#   covariates but is NOT autoregressive (e.g., MLP-based models with covariates).
# - Use ``AutoRegressiveBaseModel`` if the model is autoregressive but does NOT
#   support covariates.
# - Use ``AutoRegressiveBaseModelWithCovariates`` if the model is autoregressive and
#   supports covariates (e.g., DeepAR, Temporal Fusion Transformer).
from pytorch_forecasting.models.base import (
    BaseModel,  # or BaseModelWithCovariates, AutoRegressiveBaseModel, etc.
)

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file
# do not import the pkg class at the module level, it should
# be imported within the respective method to access that class (namely _pkg).


# todo: change class name, docstring, and select the correct base class
# (BaseModel or BaseModelWithCovariates)
class ExampleNetwork(BaseModel):
    """Custom forecasting model.

    todo: write docstring, describe your custom forecaster here

    Parameters
    ----------
    hidden_size : int, default=16
        descriptive explanation of hidden_size
    **kwargs
        Additional keyword arguments passed to ``BaseModel.__init__``.
    """

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, hidden_size: int = 16, **kwargs):
        # save the hparams
        # you can ignore some params that are not true hyperparameters:
        #   self.save_hyperparameters(ignore=["loss"])
        self.save_hyperparameters()
        super().__init__(**kwargs)

        # IMPORTANT: the self.hparams should never be overwritten or mutated
        # for handling defaults etc, write to other attributes, e.g.,
        #   self._hidden_size = some_function(self.hparams.hidden_size)

        # todo: create any required layers after this, e.g.:
        # self.fc = torch.nn.Linear(self.hparams.hidden_size, 1)

    # implement this is mandatory
    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        # todo: update the import to use the absolute path
        # to your private package file.
        # Do NOT use relative imports.
        from extension_templates.v1.network._model_pkg import (
            ExampleNetwork_pkg,
        )

        return ExampleNetwork_pkg

    # implement this is mandatory
    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] | None = None,
        **kwargs,
    ):
        """Construct model from a TimeSeriesDataSet.

        Parameters
        ----------
        dataset : TimeSeriesDataSet
            Dataset from which to derive model parameters.
        allowed_encoder_known_variable_names : list of str or None
            Names of known variables allowed in the encoder.
        **kwargs
            Additional keyword arguments passed to the model constructor.

        Returns
        -------
        model : ExampleNetwork
            Initialized model instance.
        """
        # todo: add any dataset-derived configuration here, e.g.:
        # new_kwargs = {"n_targets": len(dataset.target_names)}
        # new_kwargs.update(kwargs)

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=(allowed_encoder_known_variable_names),
            **kwargs,
        )

    # implement this is mandatory
    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input dictionary from TimeSeriesDataSet containing tensors
            such as ``encoder_cont``, ``decoder_cont``,
            ``encoder_lengths``, ``target_scale``, etc.

        Returns
        -------
        output : dict
            Network output dictionary, created via
            ``self.to_network_output(prediction=...)``.
        """
        # todo: implement the forward loop
        raise NotImplementedError("Implement forward() in your custom model")

    # ---- optional methods below ----
    # Delete these if not needed. Only implement if your model requires
    # custom post-processing (e.g., rescaling, clipping, CDF extraction).
    #
    # def to_prediction(self, out, use_metric=True, **kwargs):
    #     """Convert raw output to point predictions (optional)."""
    #     return super().to_prediction(out, use_metric=use_metric, **kwargs)
    #
    # def to_quantiles(self, out, use_metric=True, **kwargs):
    #     """Convert raw output to quantile predictions (optional)."""
    #     return super().to_quantiles(out, use_metric=use_metric, **kwargs)
