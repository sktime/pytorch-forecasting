from lightning.pytorch import Trainer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.encoders import (
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.xlstm import xLSTMTime

num_series = 100
seq_length = 50
data_list = []
for i in range(num_series):
    x = np.arange(seq_length)
    y = np.sin(x / 5.0) + np.random.normal(scale=0.1, size=seq_length)
    category = i % 5
    static_value = np.random.rand()
    for t in range(seq_length - 1):
        data_list.append(
            {
                "series_id": i,
                "time_idx": t,
                "x": y[t],
                "y": y[t + 1],
                "category": category,
                "future_known_feature": np.cos(t / 10),
                "static_feature": static_value,
                "static_feature_cat": i % 3,
            }
        )
data_df = pd.DataFrame(data_list)
# data_df.head()

dataset = TimeSeries(
    data=data_df,
    time="time_idx",
    target="y",
    group=["series_id"],
    num=["x", "future_know_feature", "static_feature"],
    cat=["category", "static_feature_cat"],
    known=["future_known_feature"],
    unknown=["x", "category"],
    static=["static_feature", "static_feature_cat"],
)

data_module = EncoderDecoderTimeSeriesDataModule(
    time_series_dataset=dataset,
    max_encoder_length=30,
    max_prediction_length=1,
    add_relative_time_idx=True,
    target_normalizer=TorchNormalizer(),
    categorical_encoders={
        "category": NaNLabelEncoder(add_nan=True),
        "static_feature_cat": NaNLabelEncoder(add_nan=True),
    },
    scalers={
        "x": StandardScaler(),
        "future_known_feature": StandardScaler(),
        "static_feature": StandardScaler(),
    },
    batch_size=32,
)


model1 = xLSTMTime(
    loss=MAE(),
    hidden_size=128,
    xlstm_type="slstm",  # or "mlstm" for matrix LSTM variant
    num_layers=2,
    decomposition_kernel=25,
    dropout=0.1,
    input_projection_size=None,  # defaults to hidden_size if None
    logging_metrics=[MAE(), SMAPE()],
    optimizer="adam",
    optimizer_params={"lr": 1e-3},
    lr_scheduler="reduce_lr_on_plateau",
    lr_scheduler_params={
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
    },
    metadata=data_module.metadata,
)
trainer1 = Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    enable_progress_bar=True,
    enable_model_summary=True,
)


data_module.setup(stage="fit")
trainer1.fit(model1, data_module)
train_loader = data_module.train_dataloader()
x_batch, y_batch = next(iter(train_loader))
print("Input batch keys:", x_batch.keys())
print(
    "Target batch shape:",
    y_batch.shape if isinstance(y_batch, torch.Tensor) else [t.shape for t in y_batch],
)

print(f"Shape of x_batch['encoder_cont'] is {x_batch['encoder_cont'].shape}")

model1.eval()

with torch.no_grad():
    output = model1(x_batch)
    predictions = output["prediction"]

print(f"Predictions shape: {predictions.shape}")  # Should be [32, 1, 1]
print(f"Target shape: {y_batch.shape}")  # Should be [32, 1]
print(f"\nFirst 5 predictions:\n{predictions[:5].squeeze()}")
print(f"\nFirst 5 targets:\n{y_batch[:5].squeeze()}")
