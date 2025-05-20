import pickle
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from pandas.core.common import SettingWithCopyWarning

from pytorch_forecasting import (
    GroupNormalizer,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

warnings.simplefilter("error", category=SettingWithCopyWarning)


data = get_stallion_data()

data["month"] = data.date.dt.month.astype("str").astype("category")
data["log_volume"] = np.log(data.volume + 1e-8)

data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()
data["avg_volume_by_sku"] = data.groupby(
    ["time_idx", "sku"], observed=True
).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(
    ["time_idx", "agency"], observed=True
).volume.transform("mean")
# data = data[lambda x: (x.sku == data.iloc[0]["sku"]) & (x.agency == data.iloc[0]["agency"])] # noqa: E501
special_days = [
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
]
data[special_days] = (
    data[special_days].apply(lambda x: x.map({0: "", 1: x.name})).astype("category")
)

training_cutoff = data["time_idx"].max() - 6
max_encoder_length = 36
max_prediction_length = 6

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length
    // 2,  # allow encoder lengths from 0 to max_prediction_length
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={
        "special_days": special_days
    },  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus", center=False
    ),  # use softplus with beta=1.0 and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)
batch_size = 64
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0
)


# save datasets
training.save("t raining.pkl")
validation.save("validation.pkl")

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger(log_graph=True)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    # val_check_interval=20,
    # limit_val_batches=1,
    # fast_dev_run=True,
    logger=logger,
    # profiler=True,
    callbacks=[lr_logger, early_stop_callback],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# # find optimal learning rate
# # remove logging and artificial epoch size
# tft.hparams.log_interval = -1
# tft.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# # run learning rate finder
# res = Tuner(trainer).lr_find(
#     tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2 # noqa: E501
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# tft.hparams.learning_rate = res.suggestion()

# trainer.fit(
#     tft,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )

# # make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)


# tune
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
)
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)


# profile speed
# profile(
#     trainer.fit,
#     profile_fname="profile.prof",
#     model=tft,
#     period=0.001,
#     filter="pytorch_forecasting",
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
