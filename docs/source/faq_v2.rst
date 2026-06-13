.. _faq_v2:

FAQ for v2
==========

.. currentmodule:: pytorch_forecasting

Common questions and answers about the experimental v2 architecture
of pytorch-forecasting. Note that this API is still under developement
and may change without prior notice.

For general questions about v1, see :doc:`FAQ <faq>`.
Other places to seek help:

* `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io>`_ and issues
* `PyTorch documentation <https://pytorch.org/>`_ and issues
* `Stack Overflow <https://stackoverflow.com/>`_


Architecture Overview
---------------------

* **What is the architecture of v2?**

  pytorch-forecasting v2 is organized into four layers to seperate data ingestion,
  preprocessing, modelling, and workflow managment:

  * **D1 Layer (TimeSeries):** The raw data ingestion layer. Takes a pandas
    DataFrame and converts it into PyTorch tensors. It also extracts metadata
    about the columns.
  * **D2 Layer (DataModule):** The processing layer. Handles preprocessing,
    scaling, windowing into encoder/decoder sequences, and creating DataLoaders.
  * **M Layer (BaseModel):** The neural network layer. Performs the forward pass,
    training step, and generates predictions.
  * **P Layer (Package):** A high-level wrapper that coordinates the other layers
    and provides a simplified ``fit()`` and ``predict()`` API.

  .. code-block:: text

      ┌──────────────────────────────────────────────────┐
      │             P Layer (Base_pkg)                   │
      │      High-level API: fit() and predict()         │
      │                                                  │
      │   ┌──────────────┐    ┌────────────────────┐     │
      │   │  D1 Layer    │───▶│  D2 Layer          │     │
      │   │  TimeSeries  │    │  DataModule        │     │
      │   │  (Raw Data)  │    │  (Preprocessing)   │     │
      │   └──────────────┘    └────────────────────┘     │
      │                              │                   │
      │                              ▼                   │
      │                     ┌────────────────────┐       │
      │                     │  M Layer           │       │
      │                     │  BaseModel         │       │
      │                     │  (Neural Network)  │       │
      │                     └────────────────────┘       │
      └──────────────────────────────────────────────────┘


Creating datasets
-----------------

* **How do I create a TimeSeries object?**

  The :py:class:`~data.timeseries.TimeSeries` class is the D1 layer.
  It takes your pandas DataFrame and converts it to raw tensors.

  .. code-block:: python

      import pandas as pd
      from pytorch_forecasting.data import TimeSeries

      # prepare your pandas DataFrame
      data = pd.DataFrame({
          "time_idx": [0, 1, 2, 3, 4, 5] * 2,
          "group": ["A"] * 6 + ["B"] * 6,
          "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                     1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
          "feature_1": [10, 20, 30, 40, 50, 60] * 2,
          "holiday": [0, 0, 1, 0, 0, 1] * 2,
      })

      # create the TimeSeries dataset
      dataset = TimeSeries(
          data=data,
          time="time_idx",           # column with time index
          target="target",           # what to predict
          group=["group"],           # each group is a seperate time series
          known=["holiday"],         # known in the future
          unknown=["feature_1"],     # not known in the future
          num=["feature_1"],         # numerical columns
          cat=["holiday"],           # categorical columns
      )

      # access a single time series by index
      sample = dataset[0]
      print(sample["y"])   # target values as tensor
      print(sample["x"])   # feature values as tensor
      print(sample["t"])   # time index values


* **What is the difference between TimeSeries, TimeSeriesDataSet and DataModule?**

  In v1, ``TimeSeriesDataSet`` handled everthing from data loading to preprocessing
  and batching in a single monolithic class. In v2, these responsibilites are split
  into ``TimeSeries`` (D1 layer, raw data only) and ``DataModule``
  (D2 layer, preprocessing and batching).

  .. list-table:: v1 vs v2 Comparison
     :header-rows: 1
     :widths: 25 35 40

     * - Feature
       - v1: ``TimeSeriesDataSet``
       - v2: ``TimeSeries`` + ``DataModule``
     * - Data Loading
       - Built-in
       - ``TimeSeries`` (D1)
     * - Preprocessing and Scaling
       - Built-in
       - ``DataModule`` (D2)
     * - Windowing and Split
       - Built-in
       - ``DataModule`` (D2)
     * - DataLoaders
       - ``to_dataloader()``
       - ``DataModule.train_dataloader()``
     * - Separation of Concerns
       - Everything in one large class
       - Clear responsibilty per layer

  The key advantage of v2 is that each layer has exactly one job, which makes
  the codebase easier to maintain and extend.


Using DataModules
-----------------

* **How do I create and use DataModules?**

  The DataModule is the D2 layer - it takes the raw tensors from the
  ``TimeSeries`` object and prepares them for the model. Currently two
  DataModules are available:

  - ``EncoderDecoderTimeSeriesDataModule`` - for encoder-decoder style models
  - ``TslibDataModule`` - for TSLib-style models (e.g. DLinear)

  .. code-block:: python

      from pytorch_forecasting.data import TimeSeries
      from pytorch_forecasting.data.data_module import (
          EncoderDecoderTimeSeriesDataModule,
      )

      # first, create the D1 layer (TimeSeries)
      dataset = TimeSeries(
          data=data,
          time="time_idx",
          target="target",
          group=["group"],
      )

      # then create the D2 layer (DataModule)
      datamodule = EncoderDecoderTimeSeriesDataModule(
          time_series_dataset=dataset,
          max_encoder_length=30,        # how much past history to use
          max_prediction_length=6,      # how far into future to predict
          batch_size=32,
          train_val_test_split=(0.7, 0.15, 0.15),
      )

      # setup creates the internal windows and data splits
      datamodule.setup(stage="fit")

      # now you can use the DataLoaders for training
      train_loader = datamodule.train_dataloader()
      val_loader = datamodule.val_dataloader()

      # metadata is available for model initalization
      print(datamodule.metadata)


Training and Prediction
-----------------------

* **How do I use pkg classes to perform fit and predict?**

  The ``pkg`` (package) is the P layer - it wraps model, datamodule, and
  trainer configuration into a single convenient interface.

  Each model class has a ``pkg`` class property that returns the corresponding
  package class. You can instanciate it with your configs and call ``fit()``
  and ``predict()``.

  .. code-block:: python

      from pytorch_forecasting.models.dlinear import DLinear

      # DLinear.pkg returns the package class (DLinear_pkg_v2)
      # instantiate it with your configuration
      pkg = DLinear.pkg(
          model_cfg={"moving_avg": 25},
          datamodule_cfg={
              "context_length": 30,
              "prediction_length": 6,
          },
          trainer_cfg={"max_epochs": 10},
      )

      # fit the model - pass your TimeSeries object directly
      best_ckpt_path = pkg.fit(dataset, save_ckpt=True)

      # generate predictions
      predictions = pkg.predict(dataset)

      # later, you can load from a checkpoint
      pkg_loaded = DLinear.pkg(ckpt_path="checkpoints/best-model.ckpt")
      predictions = pkg_loaded.predict(new_dataset)


Extending the package
---------------------

* **How do I add a new model to the package?**

  To add a new model, use the extension templates provided in
  ``extension_templates/v2/model_simple/``:

  1. Copy ``model.py`` and ``model_pkg.py`` templates to your target directory.
  2. **Implement the Model class:** Inherit from ``BaseModel`` and implement
     ``__init__()``, ``forward()``, and ``_pkg()`` methods.
  3. **Implement the Package class:** Inherit from ``Base_pkg``, set the
     ``_tags`` dictionary describing your model's capabilites, and implement
     ``get_cls()``, ``get_datamodule_cls()``, and ``get_test_train_params()``.
  4. **Register** your model in ``pytorch_forecasting/models/__init__.py``.
  5. **Verify** your implementation by running the built-in interface checks:

  .. code-block:: python

      from pytorch_forecasting.utils._estimator_checks import check_estimator
      from pytorch_forecasting.models.my_model import MyModel

      check_estimator(MyModel)

  For detailed instructions, see the
  `extension templates README <https://github.com/sktime/pytorch-forecasting/blob/main/extension_templates/v2/README.md>`_.
