from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import EncoderNormalizer, GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.utils import profile

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
    target_normalizer=EncoderNormalizer(),
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
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# save datasets
training.save("training.pkl")
validation.save("validation.pkl")

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateLogger()

trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    early_stop_callback=early_stop_callback,
    limit_train_batches=30,
    limit_val_batches=3,
    # fast_dev_run=True,
    # logger=logger,
    # profiler=True,
    callbacks=[lr_logger],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.1,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=3,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval=10,
    log_val_interval=3,
    # reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# # find optimal learning rate
# tft.hparams.log_interval = -1
# tft.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# res = trainer.lr_find(tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2)

# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# tft.hparams.learning_rate = res.suggestion()

torch.set_num_threads(10)
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# # make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)


# # tune
# study = optimize_hyperparameters(
#     train_dataloader,
#     val_dataloader,
#     model_path="optuna_test",
#     n_trials=15,
#     max_epochs=25,
#     gradient_clip_val_range=(0.01, 1.0),
#     hidden_size_range=(8, 128),
#     hidden_continuous_size_range=(8, 128),
#     attention_head_size_range=(1, 4),
#     dropout_range=(0.1, 0.3),
#     trainer_kwargs=dict(val_check_interval=20),
#     # reduce_on_plateau_patience=2,
# )
# with open("test_study.pickle", "wb") as fout:
#     pickle.dump(study, fout)


# profile speed
# profile(
#     trainer.fit,
#     profile_fname="profile.prof",
#     model=tft,
#     period=0.001,
#     filter="pytorch_forecasting",
#     train_dataloader=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
