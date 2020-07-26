import pickle
import warnings


import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pathlib import Path
import pandas as pd
import numpy as np

from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.utils import profile

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter("error", category=SettingWithCopyWarning)


from data import get_stallion_data

data = get_stallion_data()

data["month"] = data.date.dt.month
data["log_volume"] = np.log(data.volume + 1e-8)

data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

training_cutoff = data["time_idx"].max() - 6
max_encoder_length = 12
max_prediction_length = 6

features = data.drop(["volume"], axis=1).dropna()
target = data.volume[features.index]

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx < training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=[
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ],
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["volume", "log_volume", "industry_volume", "soda_volume", "avg_max_temp"],
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=15, verbose=False, mode="min")
lr_logger = LearningRateLogger()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    early_stop_callback=early_stop_callback,
    limit_train_batches=15,
    # limit_val_batches=1,
    # fast_dev_run=True,
    # logger=logger,
    # profiler=True,
    callbacks=[lr_logger],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(log_space=True),
    log_interval=2,
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# # find optimal learning rate
# res = trainer.lr_find(
#     tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
# )
#
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()


trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)

# make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)


# tune
# study = optimize_hyperparameters(
#     train_dataloader,
#     val_dataloader,
#     model_path="optuna_test",
#     n_trials=15,
#     max_epochs=10,
#     gradient_clip_val_range=(0.01, 1.0),
#     hidden_size_range=(16, 64),
#     hidden_continuous_size_range=(8, 64),
#     attention_head_size_range=(1, 4),
#     dropout_range=(0.1, 0.3),
#     learning_rate_range=(0.001, 0.1),
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
