.. pytorch-forecasting documentation master file, created by
   sphinx-quickstart on Sun Aug 16 22:17:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/logo.svg
   :height: 120

Pytorch Forecasting aims to ease timeseries forecasting with neural networks.
It specificially provides a class to wrap timeseries datasets and a number of PyTorch models.

The timeseries dataset automates many common tasks such as

* scaling and encoding of variables
* normalizing the target variable
* efficiently converting timeseries in pandas dataframes to torch tensors
* holding information about static and time-varying variables known and unknown in the future
* holiding information about related categories (such as holidays)
* downsampling for data augmentation
* generating inference, validation and test datasets
* etc.

Model implementations go beyond soley implementing neural networks. PyTorch Forecasting includes

* Model interpretation (if supported by the architecture)
* Logging of predictions and actuals in tensorboard along with a number of metrics
* etc.

They are build on `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`_ and as such

* Can be trained on multiple GPUs out of the box
* Quickly be tested with new datasets
* etc.

To install the package, execute

.. code-block::

   pip install pytorch-forecasting

Vist :ref:`Getting started<getting-started>` to learn more about the package.

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 6

   getting-started
   tutorials
   data
   models
   metrics
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
