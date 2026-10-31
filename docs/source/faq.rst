FAQ
===

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. currentmodule:: pytorch_forecasting

Other places to look for help

* :doc:`Tutorials <tutorials_v2>`
* `PyTorch Lightning documentation <https://lightning.ai/docs/pytorch/stable/>`_
* `PyTorch documentation <https://pytorch.org/docs/stable/>`_
* `GitHub issues <https://github.com/sktime/pytorch-forecasting/issues>`_


Data
----

* **How long should the encoder and prediction lengths be?**

  .. _faq_encoder_decoder_length:

  Choose something reasonably long, but not much longer than 500 for the encoder
  and 200 for the prediction length. Longer windows make training slower. The
  right ratio between the two depends on the model.

* **Why does my series not appear in training?**

  Each series must be at least one window long, which is the encoder length plus
  the prediction length. Shorter series are skipped. Also check
  ``train_val_test_split``. Whole series are assigned to train, validation or
  test, so a given series appears in only one of them.

* **How are missing values treated?**

  NaNs in the data are not filled. Fill them before building the ``TimeSeries``,
  for example with the median and an extra categorical column that marks the
  filled rows. ``allow_missing_timesteps`` is not used for now, so gaps in the
  time index are not filled either.

* **Why do I get "can't convert np.ndarray of type numpy.object_"?**

  A categorical column holds strings. Categories are not encoded for now, so
  convert them to integer codes first, for example with
  ``df["category"].astype("category").cat.codes``.

* **How do I provide future values of known features?**

  Pass them as ``data_future`` to ``TimeSeries``. It may hold only time, group,
  weight, known and static columns.


Training
--------

* **My training seems to freeze although my CPU or GPU is at 100%. How do I fix
  this?**

  The model or the windows are probably too large. Reduce ``hidden_size`` or the
  encoder and prediction lengths. See
  :ref:`How long should the encoder and prediction lengths be? <faq_encoder_decoder_length>`

* **Why are my predictions on a different scale than my data?**

  Predictions are returned in the scale of the fitted target normalizer. The
  inverse transform is not applied yet.

* **How do I choose hyperparameters?**

  Read the model's API page to see which arguments matter. To tune them, the
  `optuna package <https://optuna.org/>`_ is a good place to start.
