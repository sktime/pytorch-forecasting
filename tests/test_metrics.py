from functools import wraps
import itertools
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics import (
    MAE,
    SMAPE,
    BetaDistributionLoss,
    ImplicitQuantileNetworkDistributionLoss,
    LogNormalDistributionLoss,
    MQF2DistributionLoss,
    MultivariateNormalDistributionLoss,
    NegativeBinomialDistributionLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.metrics.base_metrics import AggregationMetric, CompositeMetric


def test_composite_metric():
    metric1 = SMAPE()
    metric2 = MAE()
    combined_metric = 1.0 * (0.3 * metric1 + 2.0 * metric2 + metric1)
    assert isinstance(
        combined_metric, CompositeMetric
    ), "combined metric should be composite metric"

    # test repr()
    repr(combined_metric)

    # test results
    y = torch.normal(0, 1, (10, 20)).abs()
    y_pred = torch.normal(0, 1, (10, 20)).abs()

    res1 = metric1(y_pred, y)
    res2 = metric2(y_pred, y)
    combined_res = combined_metric(y_pred, y)

    assert torch.isclose(combined_res, res1 * 0.3 + res2 * 2.0 + res1)

    # test quantiles and prediction
    combined_metric.to_prediction(y_pred)
    combined_metric.to_quantiles(y_pred)


@pytest.mark.parametrize(
    "decoder_lengths,y",
    [
        (
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([[0.0, 1.0], [5.0, 1.0]]),
        ),
        (2 * torch.ones(2, dtype=torch.long), torch.tensor([[0.0, 1.0], [5.0, 1.0]])),
        (
            2 * torch.ones(2, dtype=torch.long),
            torch.tensor([[[0.0, 1.0], [1.0, 1.0]], [[5.0, 1.0], [1.0, 2.0]]]),
        ),
    ],
)
def test_aggregation_metric(decoder_lengths, y):
    y_pred = torch.tensor([[0.0, 2.0], [4.0, 3.0]])
    if (decoder_lengths != y_pred.size(-1)).any():
        y_packed = rnn.pack_padded_sequence(
            y, lengths=decoder_lengths, batch_first=True, enforce_sorted=False
        )
    else:
        y_packed = y

    # metric
    metric = AggregationMetric(MAE())
    res = metric(y_pred, y_packed)
    if (decoder_lengths == y_pred.size(-1)).all() and y.ndim == 2:
        assert torch.isclose(res, (y.mean(0) - y_pred.mean(0)).abs().mean())


def test_none_reduction():
    pred = torch.rand(20, 10)
    target = torch.rand(20, 10)

    mae = MAE(reduction="none")(pred, target)
    assert mae.size() == pred.size(), "dimension should not change if reduction is none"


@pytest.mark.parametrize(
    ["center", "transformation"],
    itertools.product(
        [True, False], ["log", "log1p", "softplus", "relu", "logit", None]
    ),
)
def test_NormalDistributionLoss(center, transformation):
    mean = 1.0
    std = 0.1
    n = 100000
    target = NormalDistributionLoss.distribution_class(loc=mean, scale=std).sample((n,))
    normalizer = TorchNormalizer(center=center, transformation=transformation)
    if transformation in ["log", "log1p", "relu", "softplus"]:
        target = target.abs()
    target = normalizer.inverse_preprocess(target)

    normalized_target = normalizer.fit_transform(target).view(1, -1)
    target_scale = normalizer.get_parameters().unsqueeze(0)
    scale = torch.ones_like(normalized_target) * normalized_target.std()
    parameters = torch.stack(
        [normalized_target, scale],
        dim=-1,
    )
    loss = NormalDistributionLoss()
    rescaled_parameters = loss.rescale_parameters(
        parameters, target_scale=target_scale, encoder=normalizer
    )
    samples = loss.sample(rescaled_parameters, 1)
    assert torch.isclose(target.mean(), samples.mean(), atol=0.1, rtol=0.5)
    if center:  # if not centered, softplus distorts std too much for testing
        assert torch.isclose(target.std(), samples.std(), atol=0.1, rtol=0.7)


@pytest.mark.parametrize(
    ["center", "transformation"],
    itertools.product(
        [True, False], ["log", "log1p", "softplus", "relu", "logit", None]
    ),
)
def test_LogNormalDistributionLoss(center, transformation):
    mean = 2.0
    std = 0.2
    n = 100000
    target = LogNormalDistributionLoss.distribution_class(loc=mean, scale=std).sample(
        (n,)
    )
    normalizer = TorchNormalizer(center=center, transformation=transformation)
    normalized_target = normalizer.fit_transform(target).view(1, -1)
    target_scale = normalizer.get_parameters().unsqueeze(0)
    scale = torch.ones_like(normalized_target) * normalized_target.std()
    parameters = torch.stack(
        [normalized_target, scale],
        dim=-1,
    )
    loss = LogNormalDistributionLoss()

    if transformation not in ["log", "log1p"]:
        with pytest.raises(AssertionError):
            rescaled_parameters = loss.rescale_parameters(
                parameters, target_scale=target_scale, encoder=normalizer
            )
    else:
        rescaled_parameters = loss.rescale_parameters(
            parameters, target_scale=target_scale, encoder=normalizer
        )
        samples = loss.sample(rescaled_parameters, 1)
        assert torch.isclose(
            torch.as_tensor(mean), samples.log().mean(), atol=0.1, rtol=0.2
        )
        if center:  # if not centered, softplus distorts std too much for testing
            assert torch.isclose(
                torch.as_tensor(std), samples.log().std(), atol=0.1, rtol=0.7
            )


@pytest.mark.parametrize(
    ["center", "transformation"],
    itertools.product(
        [True, False], ["log", "log1p", "softplus", "relu", "logit", None]
    ),
)
def test_NegativeBinomialDistributionLoss(center, transformation):
    mean = 100.0
    shape = 1.0
    n = 100000
    target = (
        NegativeBinomialDistributionLoss()
        .map_x_to_distribution(torch.tensor([mean, shape]))
        .sample((n,))
    )
    normalizer = TorchNormalizer(center=center, transformation=transformation)
    normalized_target = normalizer.fit_transform(target).view(1, -1)
    target_scale = normalizer.get_parameters().unsqueeze(0)
    parameters = torch.stack(
        [normalized_target, 1.0 * torch.ones_like(normalized_target)], dim=-1
    )
    loss = NegativeBinomialDistributionLoss()

    if center or transformation in ["logit", "log"]:
        with pytest.raises(AssertionError):
            rescaled_parameters = loss.rescale_parameters(
                parameters, target_scale=target_scale, encoder=normalizer
            )
    else:
        rescaled_parameters = loss.rescale_parameters(
            parameters, target_scale=target_scale, encoder=normalizer
        )
        samples = loss.sample(rescaled_parameters, 1)
        assert torch.isclose(target.mean(), samples.mean(), atol=0.1, rtol=0.5)
        if transformation == "log1p" and not center:
            assert torch.isclose(target.std(), samples.std(), atol=0.1, rtol=0.8)
        else:
            assert torch.isclose(target.std(), samples.std(), atol=0.1, rtol=0.5)


@pytest.mark.parametrize(
    ["center", "transformation"],
    itertools.product(
        [True, False], ["log", "log1p", "softplus", "relu", "logit", None]
    ),
)
def test_BetaDistributionLoss(center, transformation):
    initial_mean = 0.1
    initial_shape = 10
    n = 100000
    target = (
        BetaDistributionLoss()
        .map_x_to_distribution(torch.tensor([initial_mean, initial_shape]))
        .sample((n,))
    )
    normalizer = TorchNormalizer(center=center, transformation=transformation)
    normalized_target = normalizer.fit_transform(target).view(1, -1)
    target_scale = normalizer.get_parameters().unsqueeze(0)
    parameters = torch.stack(
        [normalized_target, 1.0 * torch.ones_like(normalized_target)], dim=-1
    )
    loss = BetaDistributionLoss()

    if transformation not in ["logit"] or not center:
        with pytest.raises(AssertionError):
            loss.rescale_parameters(
                parameters, target_scale=target_scale, encoder=normalizer
            )
    else:
        rescaled_parameters = loss.rescale_parameters(
            parameters, target_scale=target_scale, encoder=normalizer
        )
        samples = loss.sample(rescaled_parameters, 1)
        assert torch.isclose(
            torch.as_tensor(initial_mean), samples.mean(), atol=0.01, rtol=0.01
        )  # mean=0.1
        assert torch.isclose(
            target.std(), samples.std(), atol=0.02, rtol=0.3
        )  # std=0.09


@pytest.mark.parametrize(
    ["center", "transformation"],
    itertools.product(
        [True, False], ["log", "log1p", "softplus", "relu", "logit", None]
    ),
)
def test_MultivariateNormalDistributionLoss(center, transformation):
    normalizer = TorchNormalizer(center=center, transformation=transformation)

    mean = torch.tensor([1.0, 1.0])
    std = torch.tensor([0.2, 0.1])
    cov_factor = torch.tensor([[0.0], [0.0]])
    n = 1000000

    loss = MultivariateNormalDistributionLoss()
    target = loss.distribution_class(
        loc=mean, cov_diag=std**2, cov_factor=cov_factor
    ).sample((n,))
    target = normalizer.inverse_preprocess(target)
    target = target[:, 0]
    normalized_target = normalizer.fit_transform(target).view(1, -1)
    target_scale = normalizer.get_parameters().unsqueeze(0)
    scale = torch.ones_like(normalized_target) * normalized_target.std()
    parameters = torch.concat(
        [
            normalized_target[..., None],
            scale[..., None],
            torch.zeros((1, normalized_target.size(1), loss.rank)),
        ],
        dim=-1,
    )

    rescaled_parameters = loss.rescale_parameters(
        parameters, target_scale=target_scale, encoder=normalizer
    )
    samples = loss.sample(rescaled_parameters, 1)
    assert torch.isclose(target.mean(), samples.mean(), atol=3.0, rtol=0.5)
    if center:  # if not centered, softplus distorts std too much for testing
        assert torch.isclose(target.std(), samples.std(), atol=0.1, rtol=0.7)


def test_ImplicitQuantileNetworkDistributionLoss():
    batch_size = 3
    n_timesteps = 2
    output_size = 5

    target = torch.rand((batch_size, n_timesteps))

    normalizer = TorchNormalizer(center=True, transformation="softplus")
    normalizer.fit(target.reshape(-1))

    loss = ImplicitQuantileNetworkDistributionLoss(input_size=output_size)
    x = torch.rand((batch_size, n_timesteps, output_size))
    target_scale = torch.rand((batch_size, 2))
    pred = loss.rescale_parameters(x, target_scale=target_scale, encoder=normalizer)
    assert loss.loss(pred, target).shape == target.shape
    quantiles = loss.to_quantiles(pred)
    assert quantiles.size(-1) == len(loss.quantiles)
    assert quantiles.size(0) == batch_size
    assert quantiles.size(1) == n_timesteps

    point_prediction = loss.to_prediction(pred, n_samples=None)
    assert point_prediction.ndim == loss.to_prediction(pred, n_samples=100).ndim


@pytest.fixture
def sample_dataset():
    """Fixture to create a sample TimeSeriesDataSet for testing."""
    import numpy as np
    import pandas as pd

    rows = 15
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "label": ["test"] * rows,
            "var1": np.random.randn(rows).cumsum(),
            "var2": np.random.randn(rows).cumsum(),
        }
    )
    df = df.sort_values("time").reset_index(drop=True)
    df["past_var1"] = df["var1"].shift(-1)
    df.dropna(subset=["past_var1"], inplace=True)
    df["time_idx"] = range(len(df))
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="past_var1",
        group_ids=["label"],
        static_categoricals=["label"],
        time_varying_known_reals=["var1", "var2"],
        time_varying_unknown_reals=["past_var1"],
        max_encoder_length=5,
        max_prediction_length=2,
        categorical_encoders={"label": NaNLabelEncoder(add_nan=False)},
    )


@pytest.fixture(params=["cuda", "mps", "cpu"])
def mock_device(request):
    """Fixture to create a mock device for testing."""
    # Create a real torch.device object
    device_str = f"{request.param}:0" if request.param in ["cuda", "mps"] else "cpu"
    mock_device = torch.device(device_str)

    orig_tensor = torch.tensor
    orig_empty = torch.empty

    @wraps(orig_tensor)
    def mock_tensor(data, *args, **kwargs):
        # Force device to CPU but assign mocked device
        kwargs["device"] = "cpu"
        tensor = orig_tensor(data, *args, **kwargs)
        tensor.device = mock_device
        return tensor

    @wraps(orig_empty)
    def mock_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        tensor = orig_empty(*args, **kwargs)
        tensor.device = mock_device
        return tensor

    if request.param == "cuda":
        mock_properties = type(
            "CudaDeviceProperties",
            (),
            {
                "major": 8,
                "minor": 0,
                "name": "Mocked CUDA Device",
                "total_memory": 8 * 1024 * 1024 * 1024,
            },
        )()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda._lazy_init", return_value=None),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.Device", return_value=mock_device),
            patch("torch.cuda.get_device_properties", return_value=mock_properties),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch("torch.cuda.set_device", return_value=None),
            patch("torch.empty", new=mock_empty),
            patch("torch.tensor", new=mock_tensor),
            patch(
                "torch.Tensor.to",
                new=lambda self, device, *args, **kwargs: self.clone()
                if isinstance(device, (str, torch.device))
                and str(device).startswith("cuda")
                else self,
            ),
            patch(
                "torch.Tensor.device",
                new_callable=PropertyMock,
                return_value=mock_device,
            ),
            patch("torch.Tensor.cuda", new=lambda self, *args, **kwargs: self.clone()),
            patch("torch.nn.Module.cuda", new=lambda self, *args, **kwargs: self),
            patch("torch.nn.Module.to", new=lambda self, device, *args, **kwargs: self),
        ):
            yield "cuda"

    elif request.param == "mps":
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.backends.mps.is_built", return_value=True),
            patch("torch.empty", new=mock_empty),
            patch("torch.tensor", new=mock_tensor),
            patch(
                "torch.Tensor.to",
                new=lambda self, device, *args, **kwargs: self.clone()
                if isinstance(device, (str, torch.device))
                and str(device).startswith("mps")
                else self,
            ),
            patch(
                "torch.Tensor.device",
                new_callable=PropertyMock,
                return_value=mock_device,
            ),
            patch("torch.Tensor.cuda", new=lambda self, *args, **kwargs: self.clone()),
            patch("torch.nn.Module.cuda", new=lambda self, *args, **kwargs: self),
            patch("torch.nn.Module.to", new=lambda self, device, *args, **kwargs: self),
        ):
            yield "mps"

    else:
        yield "cpu"


def test_MQF2DistributionLoss_device_handling(mock_device):
    loss = MQF2DistributionLoss(prediction_length=2)

    assert next(loss.picnn.parameters()).device.type == mock_device

    if mock_device == "cuda":
        loss.cuda()
        assert next(loss.picnn.parameters()).device.type == "cuda"
    elif mock_device == "cpu":
        loss.cpu()
        assert next(loss.picnn.parameters()).device.type == "cpu"
    loss.to(mock_device)
    assert next(loss.picnn.parameters()).device.type == mock_device


device_params = [
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS is not available"
        ),
    ),
    "cpu",
]


@pytest.mark.parametrize("device", device_params)
def test_MQF2DistributionLoss_full_workflow(sample_dataset, device):
    """
    Test the complete workflow from training to prediction with MQF2DistributionLoss.
    """
    import lightning.pytorch as pl

    model = TemporalFusionTransformer.from_dataset(
        sample_dataset, loss=MQF2DistributionLoss(prediction_length=2)
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator=device,
        devices="auto",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        limit_val_batches=3,
    )
    dataloader = sample_dataset.to_dataloader(train=True, batch_size=4, num_workers=0)

    trainer.fit(model, dataloader)

    raw_predictions = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator=device, devices="auto", logger=False),
    )
    # Verify predictions are on correct device
    pred_device = raw_predictions.output["prediction"].device.type
    target_device = raw_predictions.x["encoder_target"].device.type
    assert pred_device == device
    assert target_device == device
    try:
        model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0)
        plot_success = True
    except RuntimeError as e:
        if "device" in str(e).lower() or "expected" in str(e).lower():
            plot_success = False
            pytest.fail(f"Device mismatch error during plotting: {e}")
        else:
            raise e
    assert plot_success, "Plotting failed due to device mismatch"


def test_MQF2DistributionLoss_device_synchronization(mock_device, sample_dataset):
    """Test that MQF2DistributionLoss components are synchronized with the device."""
    model = TemporalFusionTransformer.from_dataset(
        sample_dataset, loss=MQF2DistributionLoss(prediction_length=2)
    )
    fake_prediction = torch.randn(4, 2, 8)

    if mock_device == "cuda":
        fake_prediction = fake_prediction.cuda()
        model.loss.map_x_to_distribution(fake_prediction)
        assert next(model.loss.picnn.parameters()).device.type == "cuda"
    if mock_device == "mps":
        fake_prediction = fake_prediction.to("mps")
        model.loss.map_x_to_distribution(fake_prediction)
        assert next(model.loss.picnn.parameters()).device.type == "mps"
    if mock_device == "cpu":
        fake_prediction = fake_prediction.cpu()
        model.loss.map_x_to_distribution(fake_prediction)
        assert next(model.loss.picnn.parameters()).device.type == "cpu"
