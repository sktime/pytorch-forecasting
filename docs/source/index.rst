.. pytorch-forecasting documentation master file, created by
   sphinx-quickstart on Sun Aug 16 22:17:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorch Forecasting Documentation
==================================

.. warning::

   The v2 API is in beta and unstable. It may change without warning and is not
   ready for production use. Feedback is welcome in
   `issue 1736 <https://github.com/sktime/pytorch-forecasting/issues/1736>`_.

.. raw:: html

   <a class="github-button" href="https://github.com/sktime/pytorch-forecasting" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star sktime/pytorch-forecasting on GitHub">GitHub</a>


PyTorch Forecasting provides neural network models for time series forecasting,
for practical use and for research. It gives beginners sensible defaults and
gives experienced users full control over each step.

* :py:class:`~pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries`
  wraps a DataFrame and records the role of each column.
* Datamodules scale the data, cut windows and build the dataloaders.
* Forecasters train any model with the same ``fit`` and ``predict`` methods.
* Several forecasting architectures, listed in the :doc:`model overview <model_list>`.
* Multi-horizon losses and metrics.

The package is built on `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_,
so models train on CPUs and on one or several GPUs without code changes.

If you do not have PyTorch installed yet, follow the
:ref:`installation instructions <install>`. Otherwise install the package with

.. code-block::

   pip install pytorch-forecasting

or with conda

.. code-block::

   conda install pytorch-forecasting pytorch>=2.0.0 -c pytorch -c conda-forge

To use the MQF2 loss (multivariate quantile loss), also run

.. code-block::

   pip install pytorch-forecasting[mqf2]

:ref:`Getting started <getting-started>` shows a first model. If you have v1 code,
read :doc:`migration`.

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 6

   getting-started
   installation
   tutorials_v2
   api_v2
   model_list
   migration
   faq
   CHANGELOG


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
