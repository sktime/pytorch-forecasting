import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.tuner import Tuner
import numpy as np
import pandas as pd

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE
from pytorch_forecasting.models.duet.duet import DUETModel


class ModelSummary(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics at the end of each validation epoch."""
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
    df["data"] = df.groupby("cols")["data"].transform(lambda x: x.bfill())

    # Creating time_idx column
    df["time_idx"] = df.groupby("cols").cumcount()

    print("------------------Data Properties------------------")
    print(df.info())
    print("---------------------------------------------------")

    max_time_idx = df["time_idx"].max()

    # Training on 80% of data
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

    # {"CI": 1, "batch_size": 32, "d_ff": 128, "d_model": 128, "dropout": 0.3,
    # "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 1,
    # "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true,
    # "num_epochs": 100, "num_experts": 3, "patch_len": 48, "patience": 5,
    # "seq_len": 512}

    model = DUETModel.from_dataset(
        trainset,
        CI=1,
        d_ff=128,
        d_model=128,
        dropout=0.3,
        e_layers=1,
        factor=3,
        fc_dropout=0.1,
        k=1,
        lradj="type1",
        n_heads=1,
        norm=True,
        num_experts=3,
        patch_len=48,
        patience=5,
        learning_rate=5e-4,
        loss=MAE(),
    )

    trainer = pl.Trainer(
        max_epochs=4,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=50,  # for faster testing
        limit_val_batches=50,  # for faster testing
        fast_dev_run=True,  # for faster testing
        callbacks=[ModelSummary()],
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
