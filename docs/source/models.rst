Models
=======

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Models. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Models <models_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP and unstable. Not yet production-ready.**

.. _models:

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset for which they are destined.

PyTorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ can be used. For example, tuning of the
:py:class:`~models.temporal_fusion_transformer.TemporalFusionTransformer`
is implemented by :py:func:`~models.temporal_fusion_transformer.tuning.optimize_hyperparameters`

Architecture
------------
The v1 models in ``pytorch-forecasting`` are separated into two distinct sub-layers:

* **The M Layer (Model):** The core ``torch`` neural network implementation, inheriting from PyTorch Lightning's ``LightningModule``. This is the primary user-facing layer for building training and prediction pipelines, initialized via the ``.from_dataset()`` method. End-users should use this layer for the ML pipelines in production.
    * **Learn more:** :doc:`M Layer Documentation <m_layer>`
    * **Examples:** :doc:`v1 Tutorials <tutorials>`

* **The P Layer (Package):** An internal wrapper around the M Layer strictly for **testing framework integration**. It provides automated test fixtures and registry tags for model discovery. End-users bypass this layer entirely, though developers contributing new architectures to the library or testing their own implementation locally using the unified test framework must implement one.
    * **Learn more:** :doc:`P Layer Documentation <pkg>`

Usage
-----
PyTorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

One example of using :py:class:`~data.timeseries.TimeSeriesDataSet` and models is given below:

.. code-block:: python

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
    from lightning.pytorch.tuner import Tuner
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

    # load data
    data = ...

    # define dataset
    max_encoder_length = 36
    max_prediction_length = 6
    training_cutoff = "YYYY-MM-DD"  # day for cutoff

    training = TimeSeriesDataSet(
        data[lambda x: x.date < training_cutoff],
        time_idx= ...,
        target= ...,
        # weight="weight",
        group_ids=[ ... ],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[ ... ],
        static_reals=[ ... ],
        time_varying_known_categoricals=[ ... ],
        time_varying_known_reals=[ ... ],
        time_varying_unknown_categoricals=[ ... ],
        time_varying_unknown_reals=[ ... ],
    )

    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    # define trainer with early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[lr_logger, early_stop_callback],
    )

    # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = Tuner(trainer).lr_find(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    # fit the model
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )


Selecting an architecture
--------------------------

Criteria for selecting an architecture depend heavily on the use-case. There are multiple selection criteria
and you should take into account.

Size and type of available data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One should particularly consider five criteria.

Availability of covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _model-covariates:

If you have covariates, that is variables in addition to the target variable itself that hold information
about the target, then your case will benefit from a model that can accommodate covariates. A model that
cannot use covariates is :py:class:`~pytorch_forecasting.models.nbeats.NBeats`.

Length of timeseries
^^^^^^^^^^^^^^^^^^^^^^

The length of time series has a significant impact on which model will work well. Unfortunately,
most models are created and tested on very long timeseries while in practice short or a mix of short and long
timeseries are often encountered. A model that can leverage covariates well such as the
:py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`
will typically perform better than other models on short timeseries. It is a significant step
from short timeseries to making cold-start predictions solely based on static covariates, i.e.
making predictions without observed history. For example,
this is only supported by the
:py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`
but does not work tremendously well.


Number of timeseries and their relation to each other
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your time series are related to each other (e.g. all sales of products of the same company),
a model that can learn relations between the timeseries can improve accuracy.
Not that only :ref:`models that can process covariates <model-covariates>` can
learn relationships between different timeseries.
If the timeseries denote different entities or exhibit very similar patterns across the board,
a model such as :py:class:`~pytorch_forecasting.models.nbeats.NBeats` will not work as well.

If you have only one or very few timeseries,
they should be very long in order for a deep learning approach to work well. Consider also
more traditional approaches.

Type of prediction task
^^^^^^^^^^^^^^^^^^^^^^^^^

Not every can do regression, classification or handle multiple targets. Some are exclusively
geared towards a single task. For example, :py:class:`~pytorch_forecasting.models.nbeats.NBeats`
can only be used for regression on a single target without covariates while the
:py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer` supports
multiple targets and even heterogeneous targets where some are continuous variables and others categorical,
i.e. regression and classification at the same time. :py:class:`~pytorch_forecasting.models.deepar.DeepAR`
can handle multiple targets but only works for regression tasks.

For long forecast horizon forecasts, :py:class:`~pytorch_forecasting.models.nhits.NHiTS` is an excellent choice
as it uses interpolation capabilities.

Supporting uncertainty
~~~~~~~~~~~~~~~~~~~~~~~

Not all models support uncertainty estimation. Those that do, might do so in different fashions.
Non-parametric models provide forecasts that are not bound to a given distribution
while parametric models assume that the data follows a specific distribution.

The parametric models will be a better choice if you
know how your data (and potentially error) is distributed. However, if you are missing this information or
cannot make an educated guess that matches reality rather well, the model's uncertainty estimates will
be adversely impacted. In this case, a non-parametric model will do much better.

:py:class:`~pytorch_forecasting.models.deepar.DeepAR` is an example for a parametric model while
the :py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`
can output quantile forecasts that can fit any distribution.
Models based on normalizing flows marry the two worlds by providing a non-parametric estimate
of a full probability distribution. PyTorch Forecasting currently does not provide
support for these but
`Pyro, a package for probabilistic programming <https://pyro.ai/examples/normalizing_flows_i.html>`_ does
if you believe that your problem is uniquely suited to this solution.

Computational requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some models have simpler architectures and less parameters than others which can
lead to significantly different training times. However, this not a general rule as demonstrated
by Zhuohan et al. in `Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers
<https://arxiv.org/abs/2002.11794>`_. Because the data for a sample for timeseries models is often far smaller than it
is for computer vision or language tasks, GPUs are often underused and increasing the width of models can be an effective way
to fully use a GPU. This can increase the speed of training while also improving accuracy.
The other path to pushing utilization of a GPU up is increasing the batch size.
However, increasing the batch size can adversly affect the generalization abilities of a trained network.
Also, take into account that often computational resources are mainly necessary for inference/prediction. The upfront task of training
a models will require developer time (also expensive!) but might be only a small part of the total compuational costs over
the lifetime of a model.

The :py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer` is
a rather large model but might benefit from being trained with.
For example, :py:class:`~pytorch_forecasting.models.nbeats.NBeats` or :py:class:`~pytorch_forecasting.models.nhits.NHiTS` are
efficient models.
Autoregressive models such as :py:class:`~pytorch_forecasting.models.deepar.DeepAR` might be quick to train
but might be slow at inference time (in case of :py:class:`~pytorch_forecasting.models.deepar.DeepAR` this is
driven by sampling results probabilistically multiple times, effectively increasing the computational burden linearly with the
number of samples.

Details and available models
-------------------------------

See the API documentation for further details on M layer and P layer and the list of the models:

.. toctree::
    :maxdepth: 2

    M Layer <m_layer>
    P Layer <pkg>
