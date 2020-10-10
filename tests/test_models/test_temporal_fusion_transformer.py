import shutil
import sys

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import dataloader

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

if sys.version.startswith("3.6"):  # python 3.6 does not have nullcontext
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result


else:
    from contextlib import nullcontext


def test_integration(multiple_dataloaders_with_coveratiates, tmp_path, gpus):
    train_dataloader = multiple_dataloaders_with_coveratiates["train"]
    val_dataloader = multiple_dataloaders_with_coveratiates["val"]
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

    # check training
    logger = TensorBoardLogger(tmp_path)
    checkpoint = ModelCheckpoint(filepath=tmp_path)
    trainer = pl.Trainer(
        checkpoint_callback=checkpoint,
        max_epochs=3,
        gpus=gpus,
        weights_summary="top",
        gradient_clip_val=0.1,
        early_stop_callback=early_stop_callback,
        fast_dev_run=True,
        logger=logger,
    )
    # test monotone constraints automatically
    if "discount_in_percent" in train_dataloader.dataset.reals:
        monotone_constaints = {"discount_in_percent": +1}
        cuda_context = torch.backends.cudnn.flags(enabled=False)
    else:
        monotone_constaints = {}
        cuda_context = nullcontext()

    with cuda_context:
        if isinstance(train_dataloader.dataset.target_normalizer, NaNLabelEncoder):
            output_size = len(train_dataloader.dataset.target_normalizer.classes_)
            loss = QuantileLoss()
        else:
            output_size = 7
            loss = QuantileLoss()
        net = TemporalFusionTransformer.from_dataset(
            train_dataloader.dataset,
            learning_rate=0.15,
            hidden_size=4,
            attention_head_size=1,
            dropout=0.2,
            hidden_continuous_size=2,
            loss=loss,
            log_interval=5,
            log_val_interval=1,
            log_gradient_flow=True,
            output_size=output_size,
            monotone_constaints=monotone_constaints,
        )
        net.size()
        try:
            trainer.fit(
                net,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # check loading
            fname = f"{trainer.checkpoint_callback.dirpath}/epoch=0.ckpt"
            net = TemporalFusionTransformer.load_from_checkpoint(fname)

            # check prediction
            net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
            # check prediction on gpu
            if not (isinstance(gpus, int) and gpus == 0):
                net.to("cuda")
                net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def model(dataloaders_with_coveratiates):
    dataset = dataloaders_with_coveratiates["train"].dataset
    net = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.15,
        hidden_size=4,
        attention_head_size=1,
        dropout=0.2,
        hidden_continuous_size=2,
        loss=PoissonLoss(),
        output_size=1,
        log_interval=5,
        log_val_interval=1,
        log_gradient_flow=True,
    )
    return net


@pytest.mark.parametrize("kwargs", [dict(mode="dataframe"), dict(mode="series"), dict(mode="raw")])
def test_predict_dependency(model, dataloaders_with_coveratiates, data_with_covariates, kwargs):
    train_dataset = dataloaders_with_coveratiates["train"].dataset
    dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, data_with_covariates[lambda x: x.agency == data_with_covariates.agency.iloc[0]], predict=True
    )
    model.predict_dependency(dataset, variable="discount", values=[0.1, 0.0], **kwargs)
    model.predict_dependency(dataset, variable="agency", values=data_with_covariates.agency.unique()[:2], **kwargs)


def test_actual_vs_predicted_plot(model, dataloaders_with_coveratiates):
    y_hat, x = model.predict(dataloaders_with_coveratiates["val"], return_x=True)
    averages = model.calculate_prediction_actual_by_variable(x, y_hat)
    model.plot_prediction_actual_by_variable(averages)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="raw"),
        dict(mode="quantiles"),
        dict(return_index=True),
        dict(return_decoder_lengths=True),
        dict(return_x=True),
    ],
)
def test_prediction_with_dataloder(model, dataloaders_with_coveratiates, kwargs):
    val_dataloader = dataloaders_with_coveratiates["val"]
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)


def test_prediction_with_dataset(model, dataloaders_with_coveratiates):
    val_dataloader = dataloaders_with_coveratiates["val"]
    model.predict(val_dataloader.dataset, fast_dev_run=True)


def test_prediction_with_dataframe(model, data_with_covariates):
    model.predict(data_with_covariates, fast_dev_run=True)


def test_hyperparameter_optimization_integration(dataloaders_with_coveratiates, tmp_path):
    train_dataloader = dataloaders_with_coveratiates["train"]
    val_dataloader = dataloaders_with_coveratiates["val"]
    try:
        optimize_hyperparameters(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_path=tmp_path,
            max_epochs=1,
            n_trials=3,
            log_dir=tmp_path,
            trainer_kwargs=dict(fast_dev_run=True, limit_train_batches=5),
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
