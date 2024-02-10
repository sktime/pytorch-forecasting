import pickle
import shutil
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import pytest
import torch

from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import MultiNormalizer
from pytorch_forecasting.metrics import CrossEntropy, MQF2DistributionLoss, MultiLoss, PoissonLoss, QuantileLoss
from pytorch_forecasting.models import TSMixer

if sys.version.startswith("3.6"):  # python 3.6 does not have nullcontext
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result

else:
    from contextlib import nullcontext


def test_integration(multiple_dataloaders_with_covariates, tmp_path):
    _integration(multiple_dataloaders_with_covariates, tmp_path, trainer_kwargs=dict(accelerator="cpu"))


def _integration(dataloader, tmp_path, loss=None, trainer_kwargs=None, **kwargs):
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

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
        **trainer_kwargs
    )
    # test monotone constraints automatically
    if "discount_in_percent" in train_dataloader.dataset.reals:
        monotone_constaints = {"discount_in_percent": +1}
        cuda_context = torch.backends.cudnn.flags(enabled=False)
    else:
        monotone_constaints = {}
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
                    CrossEntropy() if isinstance(normalizer, NaNLabelEncoder) else QuantileLoss()
                    for normalizer in train_dataloader.dataset.target_normalizer.normalizers
                ]
            )
        else:
            loss = QuantileLoss()
        net = TSMixer.from_dataset(
            train_dataloader.dataset,
            hidden_size=2,
            hidden_continuous_size=2,
            attention_head_size=1,
            dropout=0.2,
            loss=loss,
            log_interval=5,
            log_val_interval=1,
            log_gradient_flow=True,
            monotone_constaints=monotone_constaints,
            **kwargs
        )
        net.size()
        try:
            trainer.fit(
                net,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            # todo: testing somehow disables grad computation even though it is explicitly turned on -
            #       loss is calculated as "grad" for MQF2
            if not isinstance(net.loss, MQF2DistributionLoss):
                test_outputs = trainer.test(net, dataloaders=test_dataloader)
                assert len(test_outputs) > 0

            # check loading
            net = TSMixer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

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
                    assert pred_len == x.shape[0], "first dimension should be prediction length"

            check(predictions.output)
            if isinstance(predictions.output, torch.Tensor):
                assert predictions.output.ndim == 2, "shape of predictions should be batch_size x timesteps"
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
    net = TSMixer.from_dataset(
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


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)
