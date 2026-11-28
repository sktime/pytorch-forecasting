import copy
import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest
from test_models.conftest import make_dataloaders
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.models import RecurrentNetwork


def _integration(
    data_with_covariates,
    tmp_path,
    cell_type="LSTM",
    data_loader_kwargs={},
    clip_target: bool = False,
    **kwargs,
):
    data_with_covariates = data_with_covariates.copy()
    if clip_target:
        data_with_covariates["target"] = data_with_covariates["volume"].clip(1e-3, 1.0)
    else:
        data_with_covariates["target"] = data_with_covariates["volume"]
    data_loader_default_kwargs = dict(
        target="target",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
    )
    data_loader_default_kwargs.update(data_loader_kwargs)
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates, **data_loader_default_kwargs
    )
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    test_dataloader = dataloaders_with_covariates["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=3,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
    )

    net = RecurrentNetwork.from_dataset(
        train_dataloader.dataset,
        cell_type=cell_type,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=5,
        **kwargs,
    )
    net.size()
    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0
        # check loading
        net = RecurrentNetwork.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cell_type": "GRU"},
        dict(
            data_loader_kwargs=dict(
                target_normalizer=GroupNormalizer(
                    groups=["agency", "sku"], center=False
                )
            ),
        ),
        dict(
            data_loader_kwargs=dict(
                lags={"volume": [2, 5]},
                target="volume",
                time_varying_unknown_reals=["volume"],
                min_encoder_length=2,
            )
        ),
        dict(
            data_loader_kwargs=dict(
                time_varying_unknown_reals=["volume", "discount"],
                target=["volume", "discount"],
                lags={"volume": [2], "discount": [2]},
            )
        ),
    ],
)
def test_integration(data_with_covariates, tmp_path, kwargs):
    _integration(data_with_covariates, tmp_path, **kwargs)


@pytest.fixture(scope="session")
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = RecurrentNetwork.from_dataset(
        dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=5,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_prediction_does_not_use_future_lagged_targets(lag):
    """In prediction mode, the lagged target at decoder step ``idx >= lag`` refers
    to a decoder target and must be filled with an earlier prediction, so the
    output must not depend on the decoder targets at all.
    """
    max_prediction_length = 5
    data = pd.DataFrame(
        dict(
            series=np.repeat(["a", "b"], 30),
            time_idx=np.tile(np.arange(30), 2),
            value=np.sin(np.arange(60) / 3.0),
        )
    )
    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        max_encoder_length=8,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["value"],
        lags={"value": [lag]},
    )
    x, _ = next(iter(dataset.to_dataloader(train=False, batch_size=4)))
    net = RecurrentNetwork.from_dataset(dataset, hidden_size=5)
    net.eval()

    # overwrite all decoder inputs that hold decoder targets: the target itself
    # and the lagged target from decoder step ``lag`` on
    x_perturbed = copy.deepcopy(x)
    x_perturbed["decoder_cont"][:, :, net.target_positions] = 100.0
    x_perturbed["decoder_cont"][:, lag:, net.lagged_target_positions[lag]] = 100.0

    with torch.no_grad():
        expected = net(x)["prediction"]
        actual = net(x_perturbed)["prediction"]
    torch.testing.assert_close(actual, expected)
