Models
======

.. _models:

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset for which they are destined.

PyTorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directy derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ can be used. For example, tuning of the
:py:class:`~models.temporal_fusion_transformer.TemporalFusionTransformer`
is implemented by :py:func:`~models.temporal_fusion_transformer.tuning.optimize_hyperparameters`

Selecting an architecture
--------------------------

Criteria for selecting an architecture depend heavily on the use-case. There are multiple selection criteria
and you should take into account. Here is an overview over the pros and cons of the implemented models:

.. csv-table:: Model comparison
   :header: "Name",                                                                                        "Covariates", "Multiple targets", "Regression", "Classification", "Probabilistic", "Uncertainty", "Interactions between series", "Flexible history length", "Cold-start", "Required computational resources (1-5, 5=most)"

   :py:class:`~pytorch_forecasting.models.rnn.RecurrentNetwork`,                                           "x",          "x",                "x",          "",               "",               "",           "",                            "x",                       "",           2
   :py:class:`~pytorch_forecasting.models.mlp.DecoderMLP`,                                                 "x",          "x",                "x",          "x",              "",               "x",          "",                            "x",                       "x",          1
   :py:class:`~pytorch_forecasting.models.nbeats.NBeats`,                                                  "",           "",                 "x",          "",               "",               "",           "",                            "",                        "",           1
   :py:class:`~pytorch_forecasting.models.deepar.DeepAR`,                                                  "x",          "x",                "x",          "",               "x",              "x",          "",                            "x",                       "",           3
   :py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`,          "x",          "x",                "x",          "x",              "",               "x",          "",                            "x",                       "x",          4


Size and type of available data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One should particularly consider five criteria.

Availability of covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _model-covariates:

If you have covariates, that is variables in addition to the target variable itself that hold information
about the target, then your case will benefit from a model that can accomodate covariates. A model that
cannot use covariates is :py:class:`~pytorch_forecasting.models.nbeats.NBeats`.

Length of timeseries
^^^^^^^^^^^^^^^^^^^^^^

The length of time series has a significant impact on which model will work well. Unfortunately,
most models are created and tested on very long timeseries while in practice short or a mix of short and long
timeseries are often encountered. A model that can leverage covariates well such as the
:py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`
will typically perform better than other models on short timeseries. It is a significant step
from short timeseries to making cold-start predictions soley based on static covariates, i.e.
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
If the timeseries denote different entities or exhibit very similar patterns accross the board,
a model such as :py:class:`~pytorch_forecasting.models.nbeats.NBeats` will work as well.

If you have only one or very few timeseries,
they should be very long in order for a deep learning approach to work well. Consider also
more traditional approaches.

Type of prediction task
^^^^^^^^^^^^^^^^^^^^^^^^^

Not every can do regression, classification or handle multiple targets. Some are exclusively
geared towards a single task. For example, :py:class:`~pytorch_forecasting.models.nbeats.NBeats`
can only be used for regression on a single target without covariates while the
:py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer` supports
multiple targets and even hetrogeneous targets where some are continuous variables and others categorical,
i.e. regression and classification at the same time. :py:class:`~pytorch_forecasting.models.deepar.DeepAR`
can handle multiple targets but only works for regression tasks.

Supporting uncertainty
~~~~~~~~~~~~~~~~~~~~~~~

Not all models support uncertainty estimation. Those that do, might do so in different fashions.
Non-parameteric models provide forecasts that are not bound to a given distribution
while parametric models assume that the data follows a specific distribution.

The parametric models will be a better choice if you
know how your data (and potentially error) is distributed. However, if you are missing this information or
cannot make an educated guess that matches reality rather well, the model's uncertainty estimates will
be adversely impacted. In this case, a non-parameteric model will do much better.

:py:class:`~pytorch_forecasting.models.deepar.DeepAR` is an example for a parameteric model while
the :py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`
can output quantile forecasts that can fit any distribution.
Models based on normalizing flows marry the two worlds by providing a non-parameteric estimate
of a full probability distribution. PyTorch Forecasting currently does not provide
support for these but
`Pyro, a package for probabilistic programming <https://pyro.ai/examples/normalizing_flows_i.html>`_ does
if you believe that your problem is uniquely suited to this solution.

Computational requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some models have simpler architectures and less parameters than others which can
lead to significantly different training times. However, this not a general rule as demonstrated
by Zhuohan et al. in `Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers
<https://arxiv.org/abs/2002.11794>`_. Because the data for a sample for timeseries models is often far samller than it
is for computer vision or language tasks, GPUs are often underused and increasing the width of models can be an effective way
to fully use a GPU. This can increase the speed of training while also improving accuracy.
The other path to pushing utilization of a GPU up is increasing the batch size.
However, increasing the batch size can adversly affect the generalization abilities of a trained network.
Also, take into account that often computational resources are mainly necessary for inference/prediction. The upfront task of training
a models will require developer time (also expensive!) but might be only a small part of the total compuational costs over
the lifetime of a model.

The :py:class:`~pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer` is
a rather large model but might benefit from being trained with.
For example, :py:class:`~pytorch_forecasting.models.nbeats.NBeats` is an efficient model.
Autoregressive models such as :py:class:`~pytorch_forecasting.models.deepar.DeepAR` might be quick to train
but might be slow at inference time (in case of :py:class:`~pytorch_forecasting.models.deepar.DeepAR` this is
driven by sampling results probabilistically multiple times, effectively increasing the computational burden linearly with the
number of samples.


Implementing new architectures
-------------------------------

Please see the :ref:`Using custom data and implementing custom models <new-model-tutorial>` tutorial on how implement basic and more advanced models.

Every model should inherit from a base model in :py:mod:`~pytorch_forecasting.models.base_model`.

.. autoclass:: pytorch_forecasting.models.base_model.BaseModel
   :noindex:
   :members: __init__



Details and available models
-------------------------------

See the API documentation for further details on available models:

.. currentmodule:: pytorch_forecasting

.. moduleautosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:

   pytorch_forecasting.models
