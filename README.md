# Timeseries forecasting with Pytorch

Install with 

`pip install pytorch-forecasting`

## Available models

* [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
* 

## Usage

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# load data
data = ... 

# define dataset
max_encode_length = 36
max_prediction_length = 6
training_cutoff = "YYYY-MM-DD"  # day for cutoff

training = TimeSeriesDataSet(
    data[lambda x: x.date < training_cutoff],
    time_idx= ...,
    target= ...,
    # weight="weight",
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


validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.data_index.time.max() + 1)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=10,
    gpus=0,
    gradient_clip_val=0.1,
    early_stop_callback=early_stop_callback,
)


tft = TemporalFusionTransformer.from_dataset(training)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)
```
