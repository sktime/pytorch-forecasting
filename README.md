![PyTorch Forecasting](./docs/source/_static/logo.svg)

[pypi-image]: https://badge.fury.io/py/pytorch-forecasting.svg
[pypi-url]: https://pypi.python.org/pypi/pytorch-forecasting
[conda-image]: https://img.shields.io/conda/v/conda-forge/pytorch-forecasting
[conda-url]: https://anaconda.org/conda-forge/pytorch-forecasting
[build-image]: https://github.com/jdb78/pytorch-forecasting/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/jdb78/pytorch-forecasting/actions/workflows/test.yml?query=branch%3Amaster
[linter-image]: https://github.com/jdb78/pytorch-forecasting/actions/workflows/code_quality.yml/badge.svg?branch=master
[linter-url]: https://github.com/jdb78/pytorch-forecasting/actions/workflows/code_quality.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/pytorch-forecasting/badge/?version=latest
[docs-url]: https://pytorch-forecasting.readthedocs.io
[coverage-image]: https://codecov.io/gh/jdb78/pytorch-forecasting/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/jdb78/pytorch-forecasting?branch=master

[![PyPI Version][pypi-image]][pypi-url] [![Conda Version][conda-image]][conda-url] [![Docs Status][docs-image]][docs-url] [![Linter Status][linter-image]][linter-url] [![Build Status][build-image]][build-url] [![Code Coverage][coverage-image]][coverage-url]

**[Documentation](https://pytorch-forecasting.readthedocs.io)** | **[Tutorials](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://pytorch-forecasting.readthedocs.io/en/latest/CHANGELOG.html)**

_PyTorch Forecasting_ is a PyTorch-based package for forecasting time series with state-of-the-art network architectures. It provides a high-level API for training networks on pandas data frames and leverages
[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) for scalable training on (multiple) GPUs, CPUs and for automatic logging.

---

Our article on [Towards Data Science](https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46) introduces the package and provides background information.

PyTorch Forecasting aims to ease state-of-the-art timeseries forecasting with neural networks for real-world cases and research alike. The goal is to provide a high-level API with maximum flexibility for professionals and reasonable defaults for beginners.
Specifically, the package provides

- A timeseries dataset class which abstracts handling variable transformations, missing values,
  randomized subsampling, multiple history lengths, etc.
- A base model class which provides basic training of timeseries models along with logging in tensorboard
  and generic visualizations such actual vs predictions and dependency plots
- Multiple neural network architectures for timeseries forecasting that have been enhanced
  for real-world deployment and come with in-built interpretation capabilities
- Multi-horizon timeseries metrics
- Ranger optimizer for faster model training
- Hyperparameter tuning with [optuna](https://optuna.readthedocs.io/)

The package is built on [pytorch-lightning](https://pytorch-lightning.readthedocs.io/) to allow training on CPUs, single and multiple GPUs out-of-the-box.

# Installation

If you are working on windows, you need to first install PyTorch with

`pip install torch -f https://download.pytorch.org/whl/torch_stable.html`.

Otherwise, you can proceed with

`pip install pytorch-forecasting`

Alternatively, you can install the package via conda

`conda install pytorch-forecasting pytorch -c pytorch>=1.7 -c conda-forge`

PyTorch Forecasting is now installed from the conda-forge channel while PyTorch is install from the pytorch channel.

To use the MQF2 loss (multivariate quantile loss), also install
`pip install pytorch-forecasting[mqf2]`

# Documentation

Visit [https://pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io) to read the
documentation with detailed tutorials.

# Available models

The documentation provides a [comparison of available models](https://pytorch-forecasting.readthedocs.io/en/latest/models.html).

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
  which outperforms DeepAR by Amazon by 36-69% in benchmarks
- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](http://arxiv.org/abs/1905.10437)
  which has (if used as ensemble) outperformed all other methods including ensembles of traditional statical
  methods in the M4 competition. The M4 competition is arguably the most important benchmark for univariate time series forecasting.
- [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](http://arxiv.org/abs/2201.12886) which supports covariates and has consistently beaten N-BEATS. It is also particularly well-suited for long-horizon forecasting.
- [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
  which is the one of the most popular forecasting algorithms and is often used as a baseline
- Simple standard networks for baselining: LSTM and GRU networks as well as a MLP on the decoder
- A baseline model that always predicts the latest known value

To implement new models or other custom components, see the [How to implement new models tutorial](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html). It covers basic as well as advanced architectures.

# Usage example

Networks can be trained with the [PyTorch Lighning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) on [pandas Dataframes](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe) which are first converted to a [TimeSeriesDataSet](https://pytorch-forecasting.readthedocs.io/en/latest/data.html).

```python
# imports for training
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

# load data: this is pandas dataframe with at least a column for
# * the target (what you want to predict)
# * the timeseries ID (which should be a unique string to identify each timeseries)
# * the time of the observation (which should be a monotonically increasing integer)
data = ...

# define the dataset, i.e. add metadata to pandas dataframe for the model to understand it
max_encoder_length = 36
max_prediction_length = 6
training_cutoff = "YYYY-MM-DD"  # day for cutoff

training = TimeSeriesDataSet(
    data[lambda x: x.date <= training_cutoff],
    time_idx= ...,  # column name of time of observation
    target= ...,  # column name of target to predict
    group_ids=[ ... ],  # column name(s) for timeseries IDs
    max_encoder_length=max_encoder_length,  # how much history to use
    max_prediction_length=max_prediction_length,  # how far to predict into future
    # covariates static for a timeseries ID
    static_categoricals=[ ... ],
    static_reals=[ ... ],
    # covariates known and unknown in the future to inform prediction
    time_varying_known_categoricals=[ ... ],
    time_varying_known_reals=[ ... ],
    time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=[ ... ],
)

# create validation dataset using the same normalization techniques as for the training dataset
validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)

# convert datasets to dataloaders for training
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# create PyTorch Lighning Trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,  # run on CPU, if on multiple GPUs, use accelerator="ddp"
    gradient_clip_val=0.1,
    limit_train_batches=30,  # 30 batches per epoch
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)

# define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
tft = TemporalFusionTransformer.from_dataset(
    # dataset
    training,
    # architecture hyperparameters
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    # loss metric to optimize
    loss=QuantileLoss(),
    # logging frequency
    log_interval=2,
    # optimizer parameters
    learning_rate=0.03,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find the optimal learning rate
res = trainer.lr_find(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)
# and plot the result - always visually confirm that the suggested learning rate makes sense
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model on the data - redefine the model with the correct learning rate if necessary
trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)
```
