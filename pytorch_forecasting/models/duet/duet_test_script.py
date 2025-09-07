import sys
from typing import Tuple  # noqa: UP035

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.tuner import Tuner
import numpy as np
import pandas as pd

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE
from pytorch_forecasting.models.duet.duet import DUETModel


class CustomModelSummary(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # This hook is called after the validation loop for an epoch has finished.
        # Access the logged metrics from the trainer's logger
        # The key for the validation metrics might vary depending on how you log them.
        # Common keys are 'val_loss', 'val_acc', etc.
        metrics = trainer.callback_metrics

        print(f"\n--- Epoch {trainer.current_epoch} Summary ---")
        if metrics:
            for key, value in metrics.items():
                print(f"{key}: {value.item():.4f}")
        else:
            print("No metrics logged for this epoch.")
        print("--------------------------------------\n")


def configure_dataset() -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    df = pd.read_csv("pytorch_forecasting/models/duet/dataset/AQWan.csv")

    df.drop("name", axis=1, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["cols", "date"], inplace=True)

    df["data"] = df.groupby("cols")["data"].transform(lambda x: x.ffill())

    # It's also good practice to backfill any NaNs that might remain at the start of a
    #  series
    df["data"] = df.groupby("cols")["data"].transform(lambda x: x.bfill())
    # -----------------------------

    print("------------------Data Properties After Filling NaNs------------------")
    print(f"Remaining NaNs in 'data' column: {df['date'].isna().sum()}")
    print("----------------------------------------------------------------------")
    df["time_idx"] = df.groupby("cols").cumcount()

    print("--- Checking for Infinite values in all columns ---")
    # Select only numeric columns to check for infinity
    numeric_cols = df.select_dtypes(include=np.number).columns
    print((df[numeric_cols] == np.inf).sum())
    print((df[numeric_cols] == -np.inf).sum())
    print("-------------------------------------------------")

    print("------------------Data Properties------------------")
    print(df.info())
    print("---------------------------------------------------")

    max_time_idx = df["time_idx"].max()
    # Use 80% of the data for training
    training_cutoff = int(max_time_idx * 0.8)

    # Create the training dataset
    training_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="data",
        group_ids=["cols"],
        max_encoder_length=512,
        max_prediction_length=96,
        target_normalizer=GroupNormalizer(groups=["cols"]),
        static_categoricals=["cols"],
        time_varying_unknown_reals=["data"],
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, min_prediction_idx=training_cutoff + 1
    )

    return training_dataset, validation_dataset


def test_duet_from_dataset():
    trainset, validationset = configure_dataset()
    assert trainset is not None
    assert validationset is not None

    train_dataloader = trainset.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = validationset.to_dataloader(
        train=False, batch_size=32, num_workers=0
    )

    # print(type(train_dataloader))
    # print(type(val_dataloader))

    # for x, y in train_dataloader:
    #     print(x)
    #     print(len(x))
    #     print(y)
    #     print(len(y))
    #     break

    model = DUETModel.from_dataset(
        trainset,
        learning_rate=1e-8,
        loss=MAE(),
    )

    trainer = pl.Trainer(
        max_epochs=10,  # Set a small number of epochs for testing
        accelerator="auto",  # Automatically uses GPU if available
        gradient_clip_val=0.1,
        # Use a smaller subset of batches for faster testing
        limit_train_batches=50,
        limit_val_batches=50,
        fast_dev_run=True,
        callbacks=[CustomModelSummary()],
    )

    # Find optimal learning rate
    Tuner(trainer).lr_find(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-10,
        max_lr=1e-1,
    )

    # Start the training
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    test_duet_from_dataset()
