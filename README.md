![](./docs/source/_static/logo.svg)

Our article on [Towards Data Science](https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46)
introduces the package and provides background information.

Pytorch Forecasting aims to ease state-of-the-art timeseries forecasting with neural networks for real-world cases and research alike. The goal is to provide a high-level API with maximum flexibility for professionals and reasonable defaults for beginners.
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

The package is built on [pytorch-lightning](https://pytorch-lightning.readthedocs.io/) to allow training on CPUs,
single and multiple GPUs out-of-the-box.

# Installation

If you are working on windows, you need to first install PyTorch with

`pip install torch -f https://download.pytorch.org/whl/torch_stable.html`.

Otherwise, you can proceed with

`pip install pytorch-forecasting`

Alternatively, you can install the package via conda

`conda install pytorch-forecasting pytorch -c pytorch>=1.7 -c conda-forge`

PyTorch Forecasting is now installed from the conda-forge channel while PyTorch is install from the pytorch channel.

# Documentation

Visit [https://pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io) to read the
documentation with detailed tutorials.

# Available models

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
  which outperforms DeepAR by Amazon by 36-69% in benchmarks
- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](http://arxiv.org/abs/1905.10437)
  which has (if used as ensemble) outperformed all other methods including ensembles of traditional statical
  methods in the M4 competition. The M4 competition is arguably the most important benchmark for univariate time series forecasting.
- [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
  which is the one of the most popular forecasting algorithms and is often used as a baseline

To implement new models, see the [How to implement new models tutorial](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html).
It covers basic as well as advanced architectures.

# Usage

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# load data
data = ...

# define dataset
max_encode_length = 36
max_prediction_length = 6
training_cutoff = "YYYY-MM-DD"  # day for cutoff

training = TimeSeriesDataSet(
    data[lambda x: x.date <= training_cutoff],
    time_idx= ...,
    target= ...,
    group_ids=[ ... ],
    max_encode_length=max_encode_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[ ... ],
    static_reals=[ ... ],
    time_varying_known_categoricals=[ ... ],
    time_varying_known_reals=[ ... ],
    time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=[ ... ],
)


validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)


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

# find optimal learning rate
res = trainer.lr_find(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)
```
