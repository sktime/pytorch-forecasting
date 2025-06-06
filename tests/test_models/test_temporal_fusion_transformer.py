import pickle
import shutil
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import Baseline, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import (
    CrossEntropy,
    MQF2DistributionLoss,
    MultiLoss,
    NegativeBinomialDistributionLoss,
    PoissonLoss,
    QuantileLoss,
    TweedieLoss,
)
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from pytorch_forecasting.utils._dependencies import _get_installed_packages

if sys.version.startswith("3.6"):  # python 3.6 does not have nullcontext
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result

else:
    from contextlib import nullcontext

from test_models.conftest import make_dataloaders


def test_integration(multiple_dataloaders_with_covariates, tmp_path):
    _integration(
        multiple_dataloaders_with_covariates,
        tmp_path,
        trainer_kwargs=dict(accelerator="cpu"),
    )


def test_non_causal_attention(dataloaders_with_covariates, tmp_path):
    _integration(
        dataloaders_with_covariates,
        tmp_path,
        causal_attention=False,
        loss=TweedieLoss(),
        trainer_kwargs=dict(accelerator="cpu"),
    )


def test_distribution_loss(data_with_covariates, tmp_path):
    data_with_covariates = data_with_covariates.assign(
        volume=lambda x: x.volume.round()
    )
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )
    _integration(
        dataloaders_with_covariates,
        tmp_path,
        loss=NegativeBinomialDistributionLoss(),
    )


@pytest.mark.skipif(
    "cpflows" not in _get_installed_packages(),
    reason="Test skipped if required package cpflows not available",
)
def test_mqf2_loss(data_with_covariates, tmp_path):
    data_with_covariates = data_with_covariates.assign(
        volume=lambda x: x.volume.round()
    )
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=GroupNormalizer(
            groups=["agency", "sku"], center=False, transformation="log1p"
        ),
    )

    prediction_length = dataloaders_with_covariates[
        "train"
    ].dataset.min_prediction_length

    _integration(
        dataloaders_with_covariates,
        tmp_path,
        loss=MQF2DistributionLoss(prediction_length=prediction_length),
        learning_rate=1e-3,
        trainer_kwargs=dict(accelerator="cpu"),
    )


def _integration(dataloader, tmp_path, loss=None, trainer_kwargs=None, **kwargs):
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    # check training
    logger = TensorBoardLogger(tmp_path)
    if trainer_kwargs is None:
        trainer_kwargs = {}
    trainer = pl.Trainer(
        max_epochs=2,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
        **trainer_kwargs,
    )
    # test monotone constraints automatically
    if "discount_in_percent" in train_dataloader.dataset.reals:
        monotone_constraints = {"discount_in_percent": +1}
        cuda_context = torch.backends.cudnn.flags(enabled=False)
    else:
        monotone_constraints = {}
        cuda_context = nullcontext()

    kwargs.setdefault("learning_rate", 0.15)

    with cuda_context:
        if loss is not None:
            pass
        elif isinstance(train_dataloader.dataset.target_normalizer, NaNLabelEncoder):
            loss = CrossEntropy()
        elif isinstance(train_dataloader.dataset.target_normalizer, MultiNormalizer):
            loss = MultiLoss(
                [
                    (
                        CrossEntropy()
                        if isinstance(normalizer, NaNLabelEncoder)
                        else QuantileLoss()
                    )
                    for normalizer in train_dataloader.dataset.target_normalizer.normalizers  # noqa : E501
                ]
            )
        else:
            loss = QuantileLoss()
        net = TemporalFusionTransformer.from_dataset(
            train_dataloader.dataset,
            hidden_size=2,
            hidden_continuous_size=2,
            attention_head_size=1,
            dropout=0.2,
            loss=loss,
            log_interval=5,
            log_val_interval=1,
            log_gradient_flow=True,
            monotone_constraints=monotone_constraints,
            **kwargs,
        )
        net.size()
        try:
            trainer.fit(
                net,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            # todo: testing somehow disables grad computation
            # even though it is explicitly turned on -
            #       loss is calculated as "grad" for MQF2
            if not isinstance(net.loss, MQF2DistributionLoss):
                test_outputs = trainer.test(net, dataloaders=test_dataloader)
                assert len(test_outputs) > 0

            # check loading
            net = TemporalFusionTransformer.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

            # check prediction
            predictions = net.predict(
                val_dataloader,
                return_index=True,
                return_x=True,
                return_y=True,
                fast_dev_run=True,
                trainer_kwargs=trainer_kwargs,
            )
            pred_len = len(predictions.index)

            # check that output is of correct shape
            def check(x):
                if isinstance(x, (tuple, list)):
                    for xi in x:
                        check(xi)
                elif isinstance(x, dict):
                    for xi in x.values():
                        check(xi)
                else:
                    assert (
                        pred_len == x.shape[0]
                    ), "first dimension should be prediction length"

            check(predictions.output)
            if isinstance(predictions.output, torch.Tensor):
                assert (
                    predictions.output.ndim == 2
                ), "shape of predictions should be batch_size x timesteps"
            else:
                assert all(
                    p.ndim == 2 for p in predictions.output
                ), "shape of predictions should be batch_size x timesteps"
            check(predictions.x)
            check(predictions.index)

            # predict raw
            net.predict(
                val_dataloader,
                return_index=True,
                return_x=True,
                fast_dev_run=True,
                mode="raw",
                trainer_kwargs=trainer_kwargs,
            )

        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def model(dataloaders_with_covariates):
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
    return net


def test_tensorboard_graph_log(dataloaders_with_covariates, model, tmp_path):
    d = next(iter(dataloaders_with_covariates["train"]))
    logger = TensorBoardLogger("test", str(tmp_path), log_graph=True)
    logger.log_graph(model, d[0])


def test_init_shared_network(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = TemporalFusionTransformer.from_dataset(
        dataset, share_single_variable_networks=True
    )
    net.predict(dataset, fast_dev_run=True)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Test skipped on Windows OS due to issues with ddp, see #1623",
)
@pytest.mark.parametrize("strategy", ["ddp"])
def test_distribution(dataloaders_with_covariates, tmp_path, strategy):
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    net = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
    )
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=3,
        gradient_clip_val=0.1,
        fast_dev_run=True,
        logger=logger,
        strategy=strategy,
        enable_checkpointing=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.parametrize(
    "kwargs", [dict(mode="dataframe"), dict(mode="series"), dict(mode="raw")]
)
def test_predict_dependency(
    model, dataloaders_with_covariates, data_with_covariates, kwargs
):
    train_dataset = dataloaders_with_covariates["train"].dataset
    data_with_covariates = data_with_covariates.copy()
    dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        data_with_covariates[lambda x: x.agency == data_with_covariates.agency.iloc[0]],
        predict=True,
    )
    model.predict_dependency(dataset, variable="discount", values=[0.1, 0.0], **kwargs)
    model.predict_dependency(
        dataset,
        variable="agency",
        values=data_with_covariates.agency.unique()[:2],
        **kwargs,
    )


@pytest.mark.skipif(
    "matplotlib" not in _get_installed_packages(),
    reason="skip test if required package matplotlib not installed",
)
def test_actual_vs_predicted_plot(model, dataloaders_with_covariates):
    prediction = model.predict(dataloaders_with_covariates["val"], return_x=True)
    averages = model.calculate_prediction_actual_by_variable(
        prediction.x, prediction.output
    )
    model.plot_prediction_actual_by_variable(averages)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="raw"),
        dict(mode="quantiles"),
        dict(return_index=True),
        dict(return_decoder_lengths=True),
        dict(return_x=True),
        dict(return_y=True),
    ],
)
def test_prediction_with_dataloder(model, dataloaders_with_covariates, kwargs):
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)


def test_prediction_with_dataloder_raw(data_with_covariates, tmp_path):
    # tests correct concatenation of raw output
    test_data = data_with_covariates.copy()
    np.random.seed(2)
    test_data = test_data.sample(frac=0.5)

    dataset = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        max_encoder_length=8,
        max_prediction_length=10,
        min_prediction_length=1,
        min_encoder_length=1,
        target="volume",
        group_ids=["agency", "sku"],
        constant_fill_strategy=dict(volume=0.0),
        allow_missing_timesteps=True,
        time_varying_unknown_reals=["volume"],
        time_varying_known_reals=["time_idx"],
        target_normalizer=GroupNormalizer(groups=["agency", "sku"]),
    )

    net = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=1e-6,
        hidden_size=4,
        attention_head_size=1,
        dropout=0.2,
        hidden_continuous_size=2,
        log_interval=1,
        log_val_interval=1,
        log_gradient_flow=True,
    )
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(max_epochs=1, gradient_clip_val=1e-6, logger=logger)
    trainer.fit(
        net, train_dataloaders=dataset.to_dataloader(batch_size=4, num_workers=0)
    )

    # choose small batch size to provoke issue
    res = net.predict(dataset.to_dataloader(batch_size=2, num_workers=0), mode="raw")
    # check that interpretation works
    net.interpret_output(res)["attention"]
    assert net.interpret_output(res.iget(slice(1)))["attention"].size() == torch.Size(
        (1, net.hparams.max_encoder_length)
    )


def test_prediction_with_dataset(model, dataloaders_with_covariates):
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader.dataset, fast_dev_run=True)


def test_prediction_with_write_to_disk(model, dataloaders_with_covariates, tmp_path):
    val_dataloader = dataloaders_with_covariates["val"]
    res = model.predict(val_dataloader.dataset, fast_dev_run=True, output_dir=tmp_path)
    assert res is None, "result should be empty when writing to disk"


def test_prediction_with_dataframe(model, data_with_covariates):
    model.predict(data_with_covariates, fast_dev_run=True)


SKIP_HYPEPARAM_TEST = (
    sys.platform.startswith("win")
    # Test skipped on Windows OS due to issues with ddp, see #1632"
    or "optuna" not in _get_installed_packages()
    or "statsmodels" not in _get_installed_packages()
    # Test skipped if required package optuna or statsmodels not available
)


@pytest.mark.skipif(
    SKIP_HYPEPARAM_TEST,
    reason="Test skipped on Win due to bug #1632, or if missing required packages",
)
@pytest.mark.parametrize("use_learning_rate_finder", [True, False])
def test_hyperparameter_optimization_integration(
    dataloaders_with_covariates, tmp_path, use_learning_rate_finder
):
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    try:
        optimize_hyperparameters(
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            model_path=tmp_path,
            max_epochs=1,
            n_trials=3,
            log_dir=tmp_path,
            trainer_kwargs=dict(
                fast_dev_run=True,
                limit_train_batches=3,
                # overwrite default trainer kwargs
                enable_progress_bar=False,
            ),
            use_learning_rate_finder=use_learning_rate_finder,
            learning_rate_range=[1e-6, 1e-2],
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_no_exogenous_variable():
    data = pd.DataFrame(
        {
            "target": np.ones(1600),
            "group_id": np.repeat(np.arange(16), 100),
            "time_idx": np.tile(np.arange(100), 16),
        }
    )
    training_dataset = TimeSeriesDataSet(
        data=data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=10,
        max_prediction_length=5,
        time_varying_unknown_reals=["target"],
        time_varying_known_reals=[],
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, data, stop_randomization=True, predict=True
    )
    training_data_loader = training_dataset.to_dataloader(
        train=True, batch_size=8, num_workers=0
    )
    validation_data_loader = validation_dataset.to_dataloader(
        train=False, batch_size=8, num_workers=0
    )
    forecaster = TemporalFusionTransformer.from_dataset(
        training_dataset,
        log_interval=1,
    )
    from lightning.pytorch import Trainer

    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=8,
        limit_val_batches=8,
    )
    trainer.fit(
        forecaster,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    best_model.predict(
        validation_data_loader,
        return_x=True,
        return_y=True,
        return_index=True,
    )


def test_correct_prediction_concatenation():
    data = generate_ar_data(seasonality=10.0, timesteps=100, n_series=2, seed=42)
    data["static"] = 2
    data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
    data.head()

    # create dataset and dataloaders
    max_encoder_length = 20
    max_prediction_length = 5

    training_cutoff = data["time_idx"].max() - max_prediction_length

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        # only unknown variable is "value"
        # and N-Beats can also not take any additional variables
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    batch_size = 71
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )

    baseline_model = Baseline()
    predictions = baseline_model.predict(
        train_dataloader,
        return_x=True,
        return_y=True,
        trainer_kwargs=dict(logger=None, accelerator="cpu"),
    )

    # The predicted output and the target should have the same size.
    assert predictions.output.size() == predictions.y[0].size()
