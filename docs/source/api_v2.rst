API
===

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. currentmodule:: pytorch_forecasting

The v2 API separates data handling from the models. A model no longer depends on
one dataset class. It is built from a small ``metadata`` object, so the same model
can be trained by a forecaster, by your own Lightning code, or on data from your
own pipeline. Forecasters give every model the same ``fit`` and ``predict``
methods, so switching models means changing one class name and its arguments.

Training and prediction pass through four layers.

* D1 layer: :py:class:`~data.timeseries._timeseries_v2.TimeSeries` wraps a
  DataFrame and returns each series as tensors. See :doc:`data_v2`.
* D2 layer: a datamodule scales the data, cuts windows, splits the series and
  builds the dataloaders. It also computes the ``metadata`` the model is built
  from. See :doc:`data_v2`.
* Model layer: the network, as a Lightning ``LightningModule``. See
  :doc:`m_layer_v2`.
* Forecaster layer: builds the other three and exposes ``fit``, ``predict`` and
  checkpoint loading. See :doc:`forecaster`.

All models use the losses and metrics in :doc:`metrics`.


Feedback
--------

The v2 API is still changing. Feedback before the release helps decide what it
looks like.

* Feedback and suggestions: `API v2 issue <https://github.com/sktime/pytorch-forecasting/issues/1736>`_
* Roadmap: `pytorch-forecasting roadmap <https://github.com/sktime/pytorch-forecasting/issues/1993>`_


.. toctree::
    :maxdepth: 2

    Data <data_v2>
    Models <models_v2>
    Metrics <metrics>
    Utils <utils>
    Tutorials <tutorials_v2>
