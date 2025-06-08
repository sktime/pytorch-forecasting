from copy import deepcopy
import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import NotFittedError, check_is_fitted
import torch

from pytorch_forecasting.data import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)


@pytest.mark.parametrize(
    "data,allow_nan",
    itertools.product(
        [
            (np.array([2, 3, 4]), np.array([1, 2, 3, 5, np.nan])),
            (np.array(["a", "b", "c"]), np.array(["q", "a", "nan"])),
        ],
        [True, False],
    ),
)
def test_NaNLabelEncoder(data, allow_nan):
    fit_data, transform_data = data
    encoder = NaNLabelEncoder(warn=False, add_nan=allow_nan)
    encoder.fit(fit_data)
    assert np.array_equal(
        encoder.inverse_transform(encoder.transform(fit_data)), fit_data
    ), "Inverse transform should reverse transform"
    if not allow_nan:
        with pytest.raises(KeyError):
            encoder.transform(transform_data)
    else:
        assert (
            encoder.transform(transform_data)[0] == 0
        ), "First value should be translated to 0 if nan"
        assert (
            encoder.transform(transform_data)[-1] == 0
        ), "Last value should be translated to 0 if nan"
        assert (
            encoder.transform(fit_data)[0] > 0
        ), "First value should not be 0 if not nan"


def test_NaNLabelEncoder_add():
    encoder = NaNLabelEncoder(add_nan=False)
    encoder.fit(np.array(["a", "b", "c"]))
    encoder2 = deepcopy(encoder)
    encoder2.fit(np.array(["d"]))
    assert encoder2.transform(np.array(["a"]))[0] == 0, "a must be encoded as 0"
    assert encoder2.transform(np.array(["d"]))[0] == 3, "d must be encoded as 3"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="robust"),
        dict(method="robust", method_kwargs=dict(upper=1.0, lower=0.0)),
        dict(method="robust", data=np.random.randn(100)),
        dict(data=np.random.randn(100)),
        dict(transformation="log"),
        dict(transformation="softplus"),
        dict(transformation="log1p"),
        dict(transformation="relu"),
        dict(method="identity"),
        dict(method="identity", data=np.random.randn(100)),
        dict(center=False),
        dict(max_length=5),
        dict(data=pd.Series(np.random.randn(100))),
        dict(max_length=[1, 2]),
    ],
)
def test_EncoderNormalizer(kwargs):
    kwargs.setdefault("method", "standard")
    kwargs.setdefault("center", True)
    kwargs.setdefault("data", torch.rand(100))
    data = kwargs.pop("data")

    normalizer = EncoderNormalizer(**kwargs)

    if kwargs.get("transformation") in ["relu", "softplus", "log1p"]:
        assert (
            normalizer.inverse_transform(
                torch.as_tensor(normalizer.fit_transform(data))
            )
            >= 0
        ).all(), "Inverse transform should yield only positive values"
    else:
        assert torch.isclose(
            normalizer.inverse_transform(
                torch.as_tensor(normalizer.fit_transform(data))
            ),
            torch.as_tensor(data),
            atol=1e-5,
        ).all(), "Inverse transform should reverse transform"


@pytest.mark.parametrize(
    "kwargs,groups",
    itertools.product(
        [
            dict(method="robust"),
            dict(transformation="log"),
            dict(transformation="relu"),
            dict(center=False),
            dict(transformation="log1p"),
            dict(transformation="softplus"),
            dict(scale_by_group=True),
        ],
        [[], ["a"]],
    ),
)
def test_GroupNormalizer(kwargs, groups):
    data = pd.DataFrame(dict(a=[1, 1, 2, 2, 3], b=[1.1, 1.1, 1.0, 0.0, 1.1]))
    defaults = dict(
        method="standard", transformation=None, center=True, scale_by_group=False
    )
    defaults.update(kwargs)
    kwargs = defaults
    kwargs["groups"] = groups
    kwargs["scale_by_group"] = kwargs["scale_by_group"] and len(kwargs["groups"]) > 0

    normalizer = GroupNormalizer(**kwargs)
    encoded = normalizer.fit_transform(data["b"], data)

    test_data = dict(
        prediction=torch.tensor([encoded[0]]),
        target_scale=torch.tensor(normalizer.get_parameters([1])).unsqueeze(0),
    )

    if kwargs.get("transformation") in ["relu", "softplus", "log1p", "log"]:
        assert (
            normalizer(test_data) >= 0
        ).all(), "Inverse transform should yield only positive values"
    else:
        assert torch.isclose(
            normalizer(test_data), torch.tensor(data.b.iloc[0]), atol=1e-5
        ).all(), "Inverse transform should reverse transform"


def test_EncoderNormalizer_with_limited_history():
    data = torch.rand(100)
    normalizer = EncoderNormalizer(max_length=[1, 2]).fit(data)
    assert normalizer.center_ == data[-1]


def test_MultiNormalizer_fitted():
    data = pd.DataFrame(
        dict(
            a=[1, 1, 2, 2, 3], b=[1.1, 1.1, 1.0, 5.0, 1.1], c=[1.1, 1.1, 1.0, 5.0, 1.1]
        )
    )

    normalizer = MultiNormalizer([GroupNormalizer(groups=["a"]), TorchNormalizer()])

    with pytest.raises(NotFittedError):
        check_is_fitted(normalizer)

    normalizer.fit(data, data)

    try:
        check_is_fitted(normalizer.normalizers[0])
        check_is_fitted(normalizer.normalizers[1])
        check_is_fitted(normalizer)
    except NotFittedError:
        pytest.fail(f"{NotFittedError}")


def test_TorchNormalizer_dtype_consistency():
    """
    - Ensures that even for float64 `target_scale`, the transformation will not change the prediction dtype.
    - Ensure that target_scale will be of type float32 if method is 'identity'
    """  # noqa: E501
    parameters = torch.tensor([[[366.4587]]])
    target_scale = torch.tensor([[427875.7500, 80367.4766]], dtype=torch.float64)
    assert (
        TorchNormalizer()(dict(prediction=parameters, target_scale=target_scale)).dtype
        == torch.float32
    )
    assert (
        TorchNormalizer().transform(parameters, target_scale=target_scale).dtype
        == torch.float32
    )

    y = np.array([1, 2, 3], dtype=np.float32)
    assert (
        TorchNormalizer(method="identity").fit(y).get_parameters().dtype
        == torch.float32
    )
