Models v2
==========

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. _models-v2:

.. currentmodule:: pytorch_forecasting

The v2 API separates the time-series data, data preparation, neural network, and
forecasting interface. This lets you use a high-level forecaster for the usual
``fit`` and ``predict`` workflow, or work directly with the model and a
Lightning trainer.

V2 workflow
-----------

The workflow has four layers:

1. **TimeSeries (data container):** Holds the input data and its time-series
   metadata, such as the time, target, group, and feature columns. It does not
   perform batching or model-specific preprocessing. Point and quantile predictions from a v2
   forecaster are also returned as a ``TimeSeries`` object; its frame can be
   accessed as a pandas DataFrame.
2. **DataModule (data preparation):** Fits and applies encoders and
   normalizers, creates the train, validation, test, and prediction data
   loaders, and provides metadata needed to initialize a model. A configured
   data module can be reused with another ``TimeSeries`` by calling
   ``with_data``; this preserves the fitted transforms.
3. **M layer (model):** A PyTorch Lightning model that implements a forecasting
   architecture and its training, evaluation, and prediction steps. It
   consumes batches prepared by a compatible DataModule.
4. **Forecaster:** A high-level interface that connects a model to its
   DataModule and Lightning trainer. Call ``fit`` with a ``TimeSeries`` to
   prepare and train the model, then ``predict`` to produce forecasts. The
   forecaster returns raw model outputs as a dictionary in ``mode="raw"``. You can also use the M
   layer and DataModule directly when you need to customize the Lightning
   workflow.

For details about the data container and data modules, see
:doc:`Data v2 <data_v2>`. For the model layer and available architectures,
see :doc:`M Layer v2 <m_layer_v2>`. The stable v1 interface is documented in
:doc:`the v1 API <api>`.

Choosing an interface
---------------------

Use a model's **Forecaster** when you want a single ``fit``/``predict``
interface to manage model initialization, data-module setup, and training. Use
the **M layer** directly when you need control over the data module or
Lightning ``Trainer``. The two interfaces use the same model implementation;
the forecaster provides orchestration around it.

See the :doc:`v2 tutorials <tutorials_v2>` for examples of the forecasting
workflow.

Usage
-----

Given a training ``TimeSeries`` named ``train_data`` and a second series
named ``prediction_data`` with the same columns, fit a Temporal Fusion
Transformer and obtain point forecasts with the following calls.

.. code-block:: python

    from lightning.pytorch import Trainer
    from pytorch_forecasting.models.temporal_fusion_transformer import TFTForecaster

    forecaster = TFTForecaster(hidden_size=16)
    forecaster.fit(train_data, trainer=Trainer(max_epochs=1), save_ckpt=False)
    forecasts = forecaster.predict(prediction_data)

To train the same architecture directly, attach the training data to its
data module and use the resulting metadata to initialize the model.

.. code-block:: python

    from pytorch_forecasting.metrics import MAE

    module = TFTForecaster.get_datamodule_cls()().with_data(train_data)
    module.setup("fit")
    model = TFTForecaster.get_cls()(metadata=module.metadata, loss=MAE())
    Trainer(max_epochs=1).fit(model, datamodule=module)

Details and available models
----------------------------

.. toctree::
    :maxdepth: 2

    M Layer <m_layer_v2>
