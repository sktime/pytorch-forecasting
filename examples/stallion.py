import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from temporal_fusion_transformer_pytorch import TimeSeriesDataSet, TemporalFusionTransformer
from pathlib import Path
from leapfrog.etl import clean_column_names, optimize_memory
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger


def parse_yearmonth(df):
    return df.assign(date=lambda x: pd.to_datetime(x.yearmonth, format="%Y%m")).drop("yearmonth", axis=1)


data_path = Path("examples/data/stallion")
weather = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "weather.csv"))).set_index(["date", "agency"])
price_sales_promotion = parse_yearmonth(
    clean_column_names(pd.read_csv(data_path / "price_sales_promotion.csv")).rename(
        columns={"sales": "price_actual", "price": "price_regular", "promotions": "discount"}
    )
).set_index(["date", "sku", "agency"])
industry_volume = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "industry_volume.csv"))).set_index("date")
industry_soda_sales = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "industry_soda_sales.csv"))).set_index(
    "date"
)
historical_volume = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "historical_volume.csv")))
event_calendar = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "event_calendar.csv"))).set_index("date")
demographics = clean_column_names(pd.read_csv(data_path / "demographics.csv")).set_index("agency")

# combine the data
data = (
    historical_volume.join(industry_volume, on="date")
    .join(industry_soda_sales, on="date")
    .join(weather, on=["date", "agency"])
    .join(price_sales_promotion, on=["date", "sku", "agency"])
    .join(demographics, on="agency")
    .join(event_calendar, on="date")
    .pipe(lambda x: optimize_memory(x, unique_value_ratio=1))
    .sort_values("date")
)

# minor feature engineering: add 12 month rolling mean volume
data = data.assign(discount_in_percent=lambda x: (x.discount / x.price_regular).fillna(0) * 100)
data["month"] = data.date.dt.month
data["volume"] = np.log1p(data.volume)

data["time_idx"] = data.date.dt.year * 12 + data.date.dt.month
data["time_idx"] = data["time_idx"] - data["time_idx"].min()

training_cutoff = "2016-09-01"
max_encode_length = 36
max_prediction_length = 6

features = data.drop(["volume"], axis=1).dropna()
target = data.volume[features.index]

training = TimeSeriesDataSet(
    data[lambda x: x.date < training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    max_encode_length=max_encode_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=[],
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
    time_varying_known_reals=[
        "time_idx",
        "price_regular",
        "price_actual",
        "discount",
        "avg_population_2017",
        "avg_yearly_household_income_2017",
        "discount_in_percent",
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["volume", "industry_volume", "soda_volume", "avg_max_temp"],
    fill_stragegy={"volume": 0},
)

validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.data.__time_idx__.max() + 1)
batch_size = 64
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    early_stop_callback=early_stop_callback,
    # limit_train_batches=1,
    # limit_val_batches=1,
    # test_percent_check = 0.01,
    # fast_dev_run=True,
    # profiler=True,
    # logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(training, learning_rate=0.2, hidden_size=32, dropout=0.1)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
trainer.fit(
    tft,
    train_dataloader=training.to_dataloader(train=True, batch_size=batch_size, num_workers=1),
    val_dataloaders=validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1),
)

# log hparams
trainer.logger.experiment.add_hparams(
    {name: value for name, value in tft.hparams.items() if isinstance(value, (float, int))},
    {name: value for name, value in trainer.callback_metrics.items() if isinstance(value, (float, int))},
)

# res = trainer.lr_find(
#     tft,
#     train_dataloader=training.to_dataloader(train=True, batch_size=batch_size, num_workers=1),
#     val_dataloaders=validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1),
# )
#
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
