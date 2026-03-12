.. pytorch-forecasting documentation master file, created by
   sphinx-quickstart on Sun Aug 16 22:17:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorch Forecasting Documentation
==================================

.. raw:: html

   <a class="github-button" href="https://github.com/sktime/pytorch-forecasting" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star sktime/pytorch-forecasting on GitHub">GitHub</a>


Our article on `Towards Data Science <https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46>`_
introduces the package and provides background information.

PyTorch Forecasting aims to ease state-of-the-art
time series forecasting with neural networks for both real-world cases and
research alike. The goal is to provide a high-level API with maximum flexibility for
professionals and reasonable defaults for beginners.
Specifically, the package provides

* A time series dataset class which abstracts handling variable transformations, missing values,
  randomized subsampling, multiple history lengths, etc.
* A base model class which provides basic training of time series models along with logging in TensorBoard
  and generic visualizations such as actual vs predictions and dependency plots
* Multiple neural network architectures for time series forecasting that have been enhanced
  for real-world deployment and come with in-built interpretation capabilities
* Multi-horizon time series metrics
* Hyperparameter tuning with `optuna <https://optuna.readthedocs.io/>`_

The package is built on `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`_ to allow
training on CPUs, single and multiple GPUs out-of-the-box.

If you do not have pytorch already installed, follow the :ref:`detailed installation instructions<install>`.

Otherwise, proceed to install the package by executing

.. code-block::

   pip install pytorch-forecasting

or to install via conda

.. code-block::

   conda install pytorch-forecasting pytorch>=1.7 -c pytorch -c conda-forge

To use the MQF2 loss (multivariate quantile loss), also execute

.. code-block::

   pip install pytorch-forecasting[mqf2]

Visit :ref:`Getting started <getting-started>` to learn more about the package and detailed installation instructions.
The :ref:`Tutorials <tutorials>` section provides guidance on how to use models and implement new ones.

Key Features
------------

PyTorch Forecasting provides several tools for deep learning based time series forecasting:

* Flexible ``TimeSeriesDataSet`` class for handling time series data
* Built-in neural forecasting models such as Temporal Fusion Transformer
* Multi-horizon forecasting metrics
* Integrated hyperparameter tuning using `optuna <https://optuna.readthedocs.io/>`_
* Visualization tools for predictions and model interpretation

Quick Example
-------------

Below is a minimal example showing how to create a dataset and train a forecasting model.

.. code-block:: python

   from pytorch_forecasting import TimeSeriesDataSet
   from pytorch_forecasting.models import TemporalFusionTransformer

   # assume dataframe "data" contains time series data
   dataset = TimeSeriesDataSet(
       data,
       time_idx="time_idx",
       target="value",
       group_ids=["series"]
   )

   model = TemporalFusionTransformer.from_dataset(dataset)
   
.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 6

   getting-started
   tutorials
   data
   models
   metrics
   faq
   installation
   api
   CHANGELOG


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
