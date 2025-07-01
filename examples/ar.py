import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
import torch

from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR

# warnings.simplefilter("error", category=SettingWithCopyWarning)


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
    lags={"value": [10, 20, 30]},
    target_normalizer=GroupNormalizer(groups=["series"]),
    add_relative_time_idx=False,
    add_target_scales=True,
    randomize_length=None,
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
print(deepar.hparams.time_varying_reals_encoder)
print(deepar.hparams.time_varying_reals_decoder)
