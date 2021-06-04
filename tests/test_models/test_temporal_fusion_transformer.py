import pickle
import shutil
import sys

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import (
    CrossEntropy,
    MultiLoss,
    NegativeBinomialDistributionLoss,
    PoissonLoss,
    QuantileLoss,
)
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

if sys.version.startswith("3.6"):  # python 3.6 does not have nullcontext
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result


else:
    from contextlib import nullcontext

from test_models.conftest import make_dataloaders


def test_integration(multiple_dataloaders_with_covariates, tmp_path, gpus):
    _integration(multiple_dataloaders_with_covariates, tmp_path, gpus)


def test_distribution_loss(data_with_covariates, tmp_path, gpus):
    data_with_covariates = data_with_covariates.assign(volume=lambda x: x.volume.round())
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )
    _integration(dataloaders_with_covariates, tmp_path, gpus, loss=NegativeBinomialDistributionLoss())


def _integration(dataloader, tmp_path, gpus, loss=None):
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

    # check training
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=gpus,
        weights_summary="top",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        checkpoint_callback=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
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
        if loss is not None:
            pass
        elif isinstance(train_dataloader.dataset.target_normalizer, NaNLabelEncoder):
            loss = CrossEntropy()
        elif isinstance(train_dataloader.dataset.target_normalizer, MultiNormalizer):
            loss = MultiLoss(
                [
                    CrossEntropy() if isinstance(normalizer, NaNLabelEncoder) else QuantileLoss()
                    for normalizer in train_dataloader.dataset.target_normalizer.normalizers
                ]
            )
        else:
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
            net = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

            # check prediction
            predictions, x, index = net.predict(val_dataloader, return_index=True, return_x=True)
            pred_len = len(val_dataloader.dataset)

            # check that output is of correct shape
            def check(x):
                if isinstance(x, (tuple, list)):
                    for xi in x:
                        check(xi)
                elif isinstance(x, dict):
                    for xi in x.values():
                        check(xi)
                else:
                    assert pred_len == x.shape[0], "first dimension should be prediction length"

            check(predictions)
            if isinstance(predictions, torch.Tensor):
                assert predictions.ndim == 2, "shape of predictions should be batch_size x timesteps"
            else:
                assert all(p.ndim == 2 for p in predictions), "shape of predictions should be batch_size x timesteps"
            check(x)
            check(index)

            # check prediction on gpu
            if not (isinstance(gpus, int) and gpus == 0):
                net.to("cuda")
                net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)

        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def model(dataloaders_with_covariates, gpus):
    dataset = dataloaders_with_covariates["train"].dataset
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
    if isinstance(gpus, list) and len(gpus) > 0:  # only run test on GPU
        net.to(gpus[0])
    return net


def test_tensorboard_graph_log(dataloaders_with_covariates, model, tmp_path):
    d = next(iter(dataloaders_with_covariates["train"]))
    logger = TensorBoardLogger("test", str(tmp_path), log_graph=True)
    logger.log_graph(model, d[0])


def test_init_shared_network(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = TemporalFusionTransformer.from_dataset(dataset, share_single_variable_networks=True)
    net.predict(dataset)


@pytest.mark.parametrize("accelerator", ["ddp", "dp"])
def test_distribution(dataloaders_with_covariates, tmp_path, accelerator, gpus):
    if isinstance(gpus, int) and gpus == 0:  # only run test on GPU
        return
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    net = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
    )
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=list(range(torch.cuda.device_count())),
        weights_summary="top",
        gradient_clip_val=0.1,
        fast_dev_run=True,
        logger=logger,
        accelerator=accelerator,
        checkpoint_callback=True,
    )
    try:
        trainer.fit(
            net,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)


@pytest.mark.parametrize("kwargs", [dict(mode="dataframe"), dict(mode="series"), dict(mode="raw")])
def test_predict_dependency(model, dataloaders_with_covariates, data_with_covariates, kwargs):
    train_dataset = dataloaders_with_covariates["train"].dataset
    dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, data_with_covariates[lambda x: x.agency == data_with_covariates.agency.iloc[0]], predict=True
    )
    model.predict_dependency(dataset, variable="discount", values=[0.1, 0.0], **kwargs)
    model.predict_dependency(dataset, variable="agency", values=data_with_covariates.agency.unique()[:2], **kwargs)


def test_actual_vs_predicted_plot(model, dataloaders_with_covariates):
    y_hat, x = model.predict(dataloaders_with_covariates["val"], return_x=True)
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
def test_prediction_with_dataloder(model, dataloaders_with_covariates, kwargs):
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)


def test_prediction_with_dataset(model, dataloaders_with_covariates):
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader.dataset, fast_dev_run=True)


def test_prediction_with_dataframe(model, data_with_covariates):
    model.predict(data_with_covariates, fast_dev_run=True)


@pytest.mark.parametrize("use_learning_rate_finder", [True, False])
def test_hyperparameter_optimization_integration(dataloaders_with_covariates, tmp_path, use_learning_rate_finder):
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    try:
        optimize_hyperparameters(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_path=tmp_path,
            max_epochs=1,
            n_trials=8,
            log_dir=tmp_path,
            trainer_kwargs=dict(
                fast_dev_run=True,
                limit_train_batches=5,
                # overwrite default trainer kwargs
                progress_bar_refresh_rate=20,
            ),
            use_learning_rate_finder=use_learning_rate_finder,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
