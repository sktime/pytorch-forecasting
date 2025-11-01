FAQ
====

.. currentmodule:: pytorch_forecasting

Common issues and answers. Other places to seek help from:

* :ref:`Tutorials <tutorials>`
* `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io>`_ and issues
* `PyTorch documentation <https://pytorch.org/>`_ and issues
* `Stack Overflow <https://stackoverflow.com/>`_


Creating datasets
-----------------

* **How do I create a dataset for new samples?**

  Use the :py:class:`~data.timeseries.TimeSeriesDataSet` method of your training dataset to
  create datasets on which you can run inference.

* **How long should the encoder and decoder/prediction length be?**

  .. _faq_encoder_decoder_length:

  Choose something reasonably long, but not much longer than 500 for the encoder length and
  200 for the decoder length. Consider that longer lengths increase the time it takes
  for your model to train.

  The ratio of decoder and encoder length depends on the used alogrithm.
  Look at :ref:`documentation <models>` to get clues.

* **It takes very long to create the dataset. Why is that?**

  If you set ``allow_missing_timesteps=True`` in your dataset, the creation of an index
  might take far more time as all missing values in the timeseries have to be identified.
  The algorithm might be possible to speed up but currently, it might be faster for you to
  not allow missing values and fill them yourself.


* **How are missing values treated?**

  #. Missing values between time points are either filled up with a fill
     forward or a constant fill-in strategy
  #. Missing values indicated by NaNs are a problem and
     should be filled in up-front, e.g. with the median value and another missing indicator categorical variable.
  #. Missing values in the future (out of range) are not filled in and
     simply not predicted. You have to provide values into the future.
     If those values are amongst the unknown future values, they will simply be ignored.


Training models
---------------

* **My training seems to freeze - nothing seem to be happening although my CPU/GPU is working at 100%.
  How to fix this issue?**

  Probably, your model is too big (check the number of parameters with ``model.size()`` or
  the dataset encoder and decoder length are unrealistically large. See
  :ref:`How long should the encoder and decoder/prediction length be? <faq_encoder_decoder_length>`

* **Why does the learning rate finder not finish?**

  First, ensure that the trainer does not have the keword ``fast_dev_run=True`` and
  ``limit_train_batches=...`` set. Second, use a target normalizer in your training dataset.
  Third, increase the ``early_stop_threshold`` argument
  of the ``lr_find`` method to a large number.

* **Why do I get lots of matplotlib warnings when running the learning rate finder?**

  This is because you keep on creating plots for logging but without a logger.
  Set ``log_interval=-1`` in your model to avoid this behaviour.

* **How do I choose hyperparameters?**

  Consult the :ref:`model documentation <models>` to understand which parameters
  are important and which ranges are reasonable. Choose the learning rate with
  the learning rate finder. To tune hyperparameters, the `optuna package <https://optuna.org/>`_
  is a great place to start with.


Interpreting models
-------------------

* **What interpretation is built into PyTorch Forecasting?**

  Look up the :ref:`model documentation <models>` for the model you use for model-specific interpretation.
  Further, all models come with some basic methods inherited from :py:class:`~models.base_model.BaseModel`.
