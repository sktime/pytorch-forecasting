.. pytorch-forecasting documentation master file, created by
   sphinx-quickstart on Sun Aug 16 22:17:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/logo.svg
   :height: 120

Pytorch Forecasting aims to ease timeseries forecasting with neural networks for real-world cases and 
research alike. Specifically, the package provides

* A timeseries dataset class which abstracts handling variable transformations, missing values,
  randomized subsampling, multiple history lengths, etc.
* A base model class which provides basic training of timeseries models along with logging in tensorboard
  and generic visualizations such actual vs predictions and dependency plots
* Multiple neural network architectures for timeseries forecasting that have been enhanced
  for real-world deployment and come with in-built interpretation capabilities
* Multi-horizon timeseries metrics
* Ranger optimizer for faster model training
* Hyperparameter tuning with `optuna <https://optuna.readthedocs.io/>`_

The package is built on `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`_ to allow 
training on CPUs, single and multiple GPUs out-of-the-box.

If you do not have pytorch already installed, follow the :ref:`detailed installation instructions<install>`.

Otherwise, proceed to install the package by executing

.. code-block::

   pip install pytorch-forecasting

or to install via conda

.. code-block::

   conda install pytorch-forecasting -c conda-forge

Vist :ref:`Getting started<getting-started>` to learn more about the package and detailled installation instruction.

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
   contribute
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
