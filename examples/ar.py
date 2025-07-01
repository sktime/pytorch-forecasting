import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import torch

from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR

warnings.simplefilter("error", category=SettingWithCopyWarning)


data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100)
data["static"] = "2"
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
validation = data.series.sample(20)

max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: ~x.series.isin(validation)],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    static_categoricals=["static"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["value"],
    time_varying_known_reals=["time_idx"],
    target_normalizer=GroupNormalizer(groups=["series"]),
    add_relative_time_idx=False,
    add_target_scales=True,
    randomize_length=None,
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    data[lambda x: x.series.isin(validation)],
    # predict=True,
    stop_randomization=True,
)
batch_size = 64
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0
)

# save datasets
training.save("training.pkl")
validation.save("validation.pkl")

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    limit_val_batches=3,
    # fast_dev_run=True,
    # logger=logger,
    # profiler=True,
    callbacks=[lr_logger, early_stop_callback],
)


deepar = DeepAR.from_dataset(
    training,
    learning_rate=0.1,
    hidden_size=32,
    dropout=0.1,
    loss=NormalDistributionLoss(),
    log_interval=10,
    log_val_interval=3,
    # reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {deepar.size() / 1e3:.1f}k")

# # find optimal learning rate
# deepar.hparams.log_interval = -1
# deepar.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# res = Tuner(trainer).lr_find(
#     deepar, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2 # noqa: E501
# )

# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# deepar.hparams.learning_rate = res.suggestion()

torch.set_num_threads(10)
trainer.fit(
    deepar,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# calcualte mean absolute error on validation set
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
predictions = deepar.predict(val_dataloader)
print(f"Mean absolute error of model: {(actuals - predictions).abs().mean()}")

# # plot actual vs. predictions
# raw_predictions, x = deepar.predict(val_dataloader, mode="raw", return_x=True)
# for idx in range(10):  # plot 10 examples
#     deepar.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
