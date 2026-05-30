"""
Minimal Lightning model extension template for PyTorch Forecasting (v1).

PURPOSE:
--------
This is NOT a working model.
It is a structured template to help contributors implement new v1 models
that integrate cleanly with PyTorch Forecasting's testing and API.

HOW TO USE:
-----------
- Copy this file and modify the class name.
- Implement the required methods marked below.
- Follow the comments carefully — they explain what each method should do.
"""

from typing import Any

import torch

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.models.base import BaseModel


class ExampleNetwork(BaseModel):
    """
    Minimal template model for contributors.

    TODO for contributors:
    -----------------------
    - Rename this class to match your model name.
    - Add your model hyperparameters to __init__
    - Implement forward()
    - Implement from_dataset()
    - Implement any additional methods required for your architecture
    - Put any reusable layers or submodules of the model into a ``layers/``
      folder (e.g., ``my_model/layers/``) to keep code modular and clean.
    """

    def __init__(self, hidden_size: int = 16, **kwargs):
        """
        Constructor for your model.

        DESIGN REQUIREMENTS & CONVENTIONS:
        ---------------------------------
        1. PyTorch Lightning Hyperparameter Saving:
           - We use ``self.save_hyperparameters()`` to automatically capture
             constructor arguments and write them to ``self.hparams``.
           - This enables automatic checkpoint save/restore and experiment
             logging.
           - ``self.save_hyperparameters()`` MUST be called before
             ``super().__init__(**kwargs)``.

        2. Selective Saving with ``ignore``:
           - **Not all parameters need to be saved.** Model-specific objects
             that are non-serializable or not true hyperparameters (e.g.,
             ``loss``, large objects, callables) should be excluded:

             .. code-block:: python

                 self.save_hyperparameters(ignore=["loss"])

           - Only standard configuration values (dimensions, layer counts,
             dropout rates, etc.) should be captured in ``hparams``.

        3. Accessing Parameters:
           - Saved parameters can be accessed in two equivalent ways:

             - ``self.hparams.hidden_size`` (explicit, via the hparams namespace)
             - ``self.hidden_size`` (direct, via Lightning's ``__getattr__``)

           - Both are valid. Use whichever style is consistent with your
             codebase. Direct access (``self.hidden_size``) is shorter;
             ``self.hparams.hidden_size`` makes it explicit that the value
             comes from saved hyperparameters.

        4. Read-Only ``self.hparams`` Constraint:
           - **CRITICAL**: Once constructor parameters are collected into
             ``self.hparams``, they should NEVER be modified, mutated, or
             overwritten after initialization.
           - If you need to derive or transform values, write them to private
             instance attributes prefixed with a leading underscore:

             .. code-block:: python

                 self._param_a = some_function(self.hparams.param_a)

        Parameters
        ----------
        hidden_size : int, default=16
            An example hyperparameter. Replace with your model's actual
            hyperparameters.
        **kwargs
            Additional keyword arguments passed to the parent
            ``BaseModel.__init__`` class constructor.
        """
        # todo: add any custom model hyperparameters to the constructor signature above

        # save_hyperparameters() stores __init__ args in self.hparams and
        # enables automatic checkpoint save/load.
        # It MUST be called before super().__init__().
        #
        # Use ignore=[] to exclude params that are not true hyperparameters
        # (e.g., loss objects, non-serializable args):
        #   self.save_hyperparameters(ignore=["loss"])
        #
        # After saving, access params via self.hparams.hidden_size
        # or directly via self.hidden_size (both work).
        #
        # Put any reusable layers or submodules into a ``layers/`` folder.
        self.save_hyperparameters()
        super().__init__(**kwargs)

        # todo: optional, parameter checking and default derivation logic.
        # Do NOT overwrite self.hparams.
        # Instead, write to private attributes:
        # self._hidden_size = some_function(self.hparams.hidden_size)

        # todo: define your network layers here, e.g.:
        # self.rnn = torch.nn.LSTM(
        #     input_size=self.hparams.hidden_size,
        #     hidden_size=self.hparams.hidden_size,
        # )
        # self.projection = torch.nn.Linear(
        #     self.hparams.hidden_size, 1,
        # )

    @classmethod
    def _pkg(cls):
        """
        REQUIRED for v1 models.

        Returns the package container class that defines metadata (_tags)
        and test fixtures.

        CRITICAL DESIGN REQUIREMENTS:
        - The package file must be in the same folder/directory as this model file.
        - The filename of the package module must be private (prefixed with an
          underscore) and match the package class name (e.g.,
          `_ExampleNetwork_pkg.py` for `ExampleNetwork_pkg`).
        - The import MUST use the absolute, fully qualified path (do NOT use
          relative imports like `from .package import ...`).
        - Example:
          ``from pytorch_forecasting.models.examplenetwork._examplenetwork_pkg``
          ``import ExampleNetwork_pkg``

        Returns
        -------
        class
            The package container class for this model.
        """
        # todo: update the import to use the absolute path to your private package file.
        # Remember: DO NOT use relative imports in PyTorch Forecasting v1 models.
        from extension_templates.v1_network_template._examplenetwork_pkg import (
            ExampleNetwork_pkg,
        )

        return ExampleNetwork_pkg

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] | None = None,
        **kwargs,
    ):
        """
        REQUIRED factory method to construct model from a TimeSeriesDataSet.

        What you should do here:
        ------------------------
        - Extract needed information from ``dataset``
          (e.g., number of targets, encoder/decoder lengths)
        - Possibly modify kwargs (e.g., set loss, logging metrics, etc.)
        - Then call super().from_dataset()

        This ensures your model is correctly initialized from data.

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
        # new_kwargs = {
        #     "n_targets": len(dataset.target_names),
        # }
        # new_kwargs.update(kwargs)

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **kwargs,
        )

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        REQUIRED: implement the forward pass of your network.

        INPUT:
        ------
        x is a dictionary from TimeSeriesDataSet containing tensors such as:
        - x["encoder_cont"], x["encoder_cat"]
        - x["decoder_cont"], x["decoder_cat"]
        - x["encoder_lengths"], x["decoder_lengths"]
        - x["target_scale"], etc.

        WHAT YOU SHOULD DO HERE:
        ------------------------
        1) Encode past sequence (optional but common)
        2) Decode future sequence
        3) Produce predictions

        OUTPUT:
        -------
        You MUST return a dictionary created via:
            return self.to_network_output(prediction=your_prediction_tensor)

        The shape of prediction should typically be:
        (batch_size, decoder_length, target_dim)

        Parameters
        ----------
        x : dict of str to torch.Tensor
            Input dictionary provided by the dataloader.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        output : dict
            Network output dictionary, created via ``self.to_network_output``.
        """
        # todo: replace with your actual model logic
        raise NotImplementedError("Implement forward() in your custom model")

    # -------------------------------------------------------------------------
    # OPTIONAL OVERRIDES (Commented out by default)
    # -------------------------------------------------------------------------
    # These methods are OPTIONAL. The BaseModel parent class already provides
    # robust default implementations for point prediction and quantile conversion.
    #
    # Do NOT implement these methods unless your model requires custom post-
    # processing or special handling. If they are not used, REMOVE/DELETE them
    # from the final code to keep the implementation clean.
    #
    # If needed, uncomment and implement:
    #
    # def to_prediction(
    #     self, out: dict[str, Any], use_metric: bool = True, **kwargs
    # ) -> torch.Tensor:
    #     """
    #     OPTIONAL override: convert raw network output to final prediction.
    #
    #     Uncomment and implement ONLY if your model requires custom post-
    #     processing of predictions (e.g., custom rescaling, clipping, etc.).
    #
    #     - Called during inference.
    #     - Default implementation in BaseModel returns ``out.prediction``.
    #
    #     Parameters
    #     ----------
    #     out : dict
    #         Raw output from forward().
    #     use_metric : bool
    #         Whether to use metric for conversion.
    #     **kwargs
    #         Additional keyword arguments.
    #
    #     Returns
    #     -------
    #     prediction : torch.Tensor
    #         Final point predictions.
    #     """
    #     return super().to_prediction(out, use_metric=use_metric, **kwargs)
    #
    # def to_quantiles(
    #     self, out: dict[str, Any], use_metric: bool = True, **kwargs
    # ) -> torch.Tensor:
    #     """
    #     OPTIONAL override: convert raw network output to quantile predictions.
    #
    #     Uncomment and implement ONLY if your model produces probabilistic
    #     outputs that require custom quantile extraction (e.g., custom CDF/PDF).
    #
    #     Parameters
    #     ----------
    #     out : dict
    #         Raw output from forward().
    #     use_metric : bool
    #         Whether to use metric for conversion.
    #     **kwargs
    #         Additional keyword arguments.
    #
    #     Returns
    #     -------
    #     quantiles : torch.Tensor
    #         Quantile predictions.
    #     """
    #     return super().to_quantiles(out, use_metric=use_metric, **kwargs)
