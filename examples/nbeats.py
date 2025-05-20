import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import pandas as pd

from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data

sys.path.append("..")


print("load data")
data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
validation = data.series.sample(20)


max_encoder_length = 150
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx < training_cutoff],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    min_encoder_length=context_length,
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    min_prediction_length=prediction_length,
    time_varying_unknown_reals=["value"],
    randomize_length=None,
    add_relative_time_idx=False,
    add_target_scales=False,
)

validation = TimeSeriesDataSet.from_dataset(
    training, data, min_prediction_idx=training_cutoff
)
batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=2
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=2
)


early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=15,
    # limit_val_batches=1,
    # fast_dev_run=True,
    # logger=logger,
    # profiler=True,
)


net = NBeats.from_dataset(
    training,
    learning_rate=3e-2,
    log_interval=10,
    log_val_interval=1,
    log_gradient_flow=False,
    weight_decay=1e-2,
)
print(f"Number of parameters in network: {net.size() / 1e3:.1f}k")

# # find optimal learning rate
# # remove logging and artificial epoch size
# net.hparams.log_interval = -1
# net.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# # run learning rate finder
# res = Tuner(trainer).lr_find(
#     net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2 # noqa: E501
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# net.hparams.learning_rate = res.suggestion()

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
