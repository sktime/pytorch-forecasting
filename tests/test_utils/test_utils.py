"""Tests for pytorch_forecasting.utils._utils module."""

from collections import namedtuple

import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting.utils._utils import (
    OutputMixIn,
    TupleOutputMixIn,
    apply_to_list,
    autocorrelation,
    concat_sequences,
    create_mask,
    detach,
    get_embedding_size,
    groupby_apply,
    integer_histogram,
    masked_op,
    move_to_device,
    next_fast_len,
    padded_stack,
    repr_class,
    to_list,
    unpack_sequence,
    unsqueeze_like,
)


class DummyOutput(OutputMixIn, namedtuple("DummyOutput", ["prediction", "scale"])):
    """Sample namedtuple container with OutputMixIn."""


class DummyModuleWithRepr(TupleOutputMixIn):
    """Sample module with TupleOutputMixIn."""

    def __init__(self, hidden_size: int = 32, num_layers: int = 2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers


# ---------------------------------------------------------------------------
# integer_histogram tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, min_val, max_val, expected",
    [
        (torch.tensor([0, 1, 1, 2, 2, 2]), None, None, [1, 2, 3]),
        (torch.tensor([1, 1, 3]), 0, 4, [0, 2, 0, 1, 0]),
        (torch.tensor([5]), 5, 5, [1]),
        (torch.tensor([2, 4]), 1, 4, [0, 1, 0, 1]),
    ],
)
def test_integer_histogram_basic(data, min_val, max_val, expected):
    hist = integer_histogram(data, min=min_val, max=max_val)
    assert torch.equal(hist, torch.tensor(expected, dtype=torch.long))


def test_integer_histogram_multidimensional():
    data = torch.tensor([[0, 1], [2, 1]])
    hist = integer_histogram(data)
    assert torch.equal(hist, torch.tensor([1, 2, 1], dtype=torch.long))


# ---------------------------------------------------------------------------
# groupby_apply tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys, values, bins, reduction, expected_reduced",
    [
        (
            torch.tensor([0, 0, 1, 2, 2]),
            torch.tensor([2.0, 4.0, 3.0, 10.0, 20.0]),
            4,
            "mean",
            [3.0, 3.0, 15.0, 0.0],
        ),
        (
            torch.tensor([0, 0, 1, 2, 2]),
            torch.tensor([2.0, 4.0, 3.0, 10.0, 20.0]),
            4,
            "sum",
            [6.0, 3.0, 30.0, 0.0],
        ),
    ],
)
def test_groupby_apply_reduction(keys, values, bins, reduction, expected_reduced):
    reduced = groupby_apply(keys, values, bins=bins, reduction=reduction)
    assert torch.allclose(reduced, torch.tensor(expected_reduced, dtype=values.dtype))


def test_groupby_apply_with_histogram():
    keys = torch.tensor([1, 1, 3])
    values = torch.tensor([2.0, 4.0, 10.0])
    reduced, hist = groupby_apply(
        keys, values, bins=5, reduction="mean", return_histogram=True
    )
    assert torch.allclose(
        reduced, torch.tensor([0.0, 3.0, 0.0, 10.0, 0.0], dtype=values.dtype)
    )
    assert torch.equal(hist, torch.tensor([0, 2, 0, 1, 0], dtype=torch.long))


def test_groupby_apply_invalid_reduction():
    keys = torch.tensor([0, 1])
    values = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="Unknown reduction 'invalid'"):
        groupby_apply(keys, values, reduction="invalid")


# ---------------------------------------------------------------------------
# get_embedding_size tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, max_size, expected",
    [
        (1, 100, 1),
        (2, 100, 1),
        (3, 100, 3),
        (10, 100, 6),
        (100, 100, 21),
        (1000, 16, 16),
        (500, 20, 20),
    ],
)
def test_get_embedding_size(n, max_size, expected):
    size = get_embedding_size(n, max_size=max_size)
    assert size == expected


# ---------------------------------------------------------------------------
# create_mask tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, lengths, inverse, expected",
    [
        (
            4,
            torch.tensor([1, 3, 0]),
            False,
            [
                [False, True, True, True],
                [False, False, False, True],
                [True, True, True, True],
            ],
        ),
        (
            4,
            torch.tensor([1, 3, 0]),
            True,
            [
                [True, False, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ],
        ),
    ],
)
def test_create_mask(size, lengths, inverse, expected):
    mask = create_mask(size, lengths, inverse=inverse)
    assert torch.equal(mask, torch.tensor(expected, dtype=torch.bool))


# ---------------------------------------------------------------------------
# next_fast_len tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, expected",
    [
        (1, 1),
        (7, 8),
        (11, 12),
        (13, 15),
        (17, 18),
        (19, 20),
        (29, 30),
    ],
)
def test_next_fast_len(size, expected):
    res = next_fast_len(size)
    assert res == expected


# ---------------------------------------------------------------------------
# autocorrelation tests
# ---------------------------------------------------------------------------


def test_autocorrelation_constant():
    x = torch.ones(2, 16)
    ac = autocorrelation(x, dim=-1)
    assert torch.all(torch.isnan(ac) | (ac == 0.0))


def test_autocorrelation_white_noise():
    torch.manual_seed(42)
    x = torch.randn(1, 500)
    ac = autocorrelation(x, dim=1)
    assert torch.isclose(ac[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.abs(ac[0, 1:]).mean() < 0.15


# ---------------------------------------------------------------------------
# unpack_sequence & concat_sequences tests
# ---------------------------------------------------------------------------


def test_unpack_sequence_tensor():
    x = torch.randn(3, 5, 2)
    seq, lengths = unpack_sequence(x)
    assert torch.equal(seq, x)
    assert torch.equal(lengths, torch.tensor([5, 5, 5], dtype=torch.long))


def test_unpack_sequence_packed():
    padded = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [0.0], [0.0]]])
    lengths = torch.tensor([3, 1])
    packed = rnn.pack_padded_sequence(
        padded, lengths=lengths, batch_first=True, enforce_sorted=False
    )
    unpacked, unpacked_lens = unpack_sequence(packed)
    assert torch.equal(unpacked, padded)
    assert torch.equal(unpacked_lens, lengths)


def test_concat_sequences_tensors():
    x1 = torch.randn(2, 4, 3)
    x2 = torch.randn(3, 4, 3)
    out = concat_sequences([x1, x2])
    assert out.shape == (5, 4, 3)
    assert torch.equal(out[:2], x1)
    assert torch.equal(out[2:], x2)


def test_concat_sequences_list_of_tensors():
    x1 = torch.tensor([1.0, 2.0])
    x2 = torch.tensor([3.0])
    out = concat_sequences([x1, x2])
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))


def test_concat_sequences_nested_tuple():
    x1 = (torch.randn(2, 3), torch.randn(2, 4))
    x2 = (torch.randn(3, 3), torch.randn(3, 4))
    out = concat_sequences([x1, x2])
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (5, 3)
    assert out[1].shape == (5, 4)


def test_concat_sequences_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported sequence type"):
        concat_sequences(["unsupported"])


# ---------------------------------------------------------------------------
# padded_stack tests
# ---------------------------------------------------------------------------


def test_padded_stack_right():
    t1 = torch.tensor([[1.0, 2.0]])
    t2 = torch.tensor([[3.0, 4.0, 5.0]])
    stacked = padded_stack([t1, t2], side="right", value=-1.0)
    expected = torch.tensor(
        [[[1.0, 2.0, -1.0]], [[3.0, 4.0, 5.0]]], dtype=torch.float32
    )
    assert torch.equal(stacked, expected)


def test_padded_stack_left():
    t1 = torch.tensor([[1.0, 2.0]])
    t2 = torch.tensor([[3.0, 4.0, 5.0]])
    stacked = padded_stack([t1, t2], side="left", value=0.0)
    expected = torch.tensor([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]], dtype=torch.float32)
    assert torch.equal(stacked, expected)


def test_padded_stack_invalid_side():
    t1 = torch.tensor([[1.0]])
    t2 = torch.tensor([[2.0, 3.0]])
    with pytest.raises(ValueError, match="side for padding 'center' is unknown"):
        padded_stack([t1, t2], side="center")


# ---------------------------------------------------------------------------
# to_list, unsqueeze_like, apply_to_list tests
# ---------------------------------------------------------------------------


def test_to_list():
    assert to_list(5) == [5]
    assert to_list([1, 2]) == [1, 2]
    assert to_list((1, 2)) == (1, 2)


def test_unsqueeze_like():
    x = torch.ones(2, 3)
    target = torch.ones(2, 3, 4, 5)
    unsqueezed = unsqueeze_like(x, target)
    assert unsqueezed.shape == (2, 3, 1, 1)

    same = unsqueeze_like(x, torch.ones(2, 3))
    assert same.shape == (2, 3)

    with pytest.raises(ValueError, match="tensor.ndim=2 > like.ndim=1"):
        unsqueeze_like(x, torch.ones(2))


def test_apply_to_list():
    assert apply_to_list([1, 2, 3], lambda v: v * 2) == [2, 4, 6]
    assert apply_to_list((1, 2), lambda v: v + 10) == [11, 12]
    assert apply_to_list(5, lambda v: v * 3) == 15


# ---------------------------------------------------------------------------
# OutputMixIn & TupleOutputMixIn tests
# ---------------------------------------------------------------------------


def test_output_mixin():
    pred = torch.tensor([1.0, 2.0])
    scale = torch.tensor([0.5, 0.5])
    out = DummyOutput(prediction=pred, scale=scale)

    assert torch.equal(out["prediction"], pred)
    assert torch.equal(out[0], pred)
    assert torch.equal(out.get("prediction"), pred)
    assert out.get("missing", 42) == 42
    assert list(out.keys()) == ["prediction", "scale"]
    items = dict(out.items())
    assert torch.equal(items["prediction"], pred)

    sliced = out.iget(slice(0, 1))
    assert isinstance(sliced, DummyOutput)
    assert torch.equal(sliced.prediction, torch.tensor([1.0]))
    assert torch.equal(sliced.scale, torch.tensor([0.5]))


def test_tuple_output_mixin():
    module = DummyModuleWithRepr()
    res1 = module.to_network_output(a=torch.tensor([1]), b=torch.tensor([2]))
    assert isinstance(res1, OutputMixIn)
    assert res1["a"].item() == 1
    assert res1.b.item() == 2

    # Cached _output_class reuse
    res2 = module.to_network_output(a=torch.tensor([3]), b=torch.tensor([4]))
    assert type(res1) is type(res2)


# ---------------------------------------------------------------------------
# move_to_device & detach tests
# ---------------------------------------------------------------------------


def test_move_to_device_cpu():
    tensor = torch.tensor([1.0, 2.0])
    container = {
        "t": tensor,
        "list": [tensor],
        "tuple": (tensor,),
        "output": DummyOutput(prediction=tensor, scale=tensor),
    }
    moved = move_to_device(container, device="cpu")
    assert moved["t"].device == torch.device("cpu")
    assert moved["list"][0].device == torch.device("cpu")
    assert moved["output"].prediction.device == torch.device("cpu")


def test_detach():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    detached = detach(y)
    assert not detached.requires_grad
    assert torch.equal(detached, y)

    nested = {"a": y, "b": [y], "c": DummyOutput(prediction=y, scale=y)}
    d_nested = detach(nested)
    assert not d_nested["a"].requires_grad
    assert not d_nested["b"][0].requires_grad
    assert not d_nested["c"].prediction.requires_grad


# ---------------------------------------------------------------------------
# masked_op tests
# ---------------------------------------------------------------------------


def test_masked_op_default_nan_mask():
    tensor = torch.tensor([[1.0, float("nan")], [3.0, 5.0]])
    mean_res = masked_op(tensor, op="mean", dim=0)
    # col 0: (1+3)/2 = 2.0, col 1: 5.0/1 = 5.0
    assert torch.allclose(mean_res, torch.tensor([2.0, 5.0]))

    sum_res = masked_op(tensor, op="sum", dim=0)
    assert torch.allclose(sum_res, torch.tensor([4.0, 5.0]))


def test_masked_op_custom_mask():
    tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    mask = torch.tensor([[True, False], [True, True]])
    mean_res = masked_op(tensor, op="mean", dim=0, mask=mask)
    # col 0: (10+30)/2 = 20.0, col 1: 40.0/1 = 40.0
    assert torch.allclose(mean_res, torch.tensor([20.0, 40.0]))


def test_masked_op_invalid_op():
    tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="unknown operation median"):
        masked_op(tensor, op="median")


# ---------------------------------------------------------------------------
# repr_class tests
# ---------------------------------------------------------------------------


def test_repr_class():
    class Sample:
        def __init__(self, alpha=1, beta="test"):
            self.alpha = alpha
            self.beta = beta

    obj = Sample()
    formatted = repr_class(obj, attributes=["alpha", "beta"])
    assert "Sample(alpha=1, beta='test')" in formatted

    with_extra = repr_class(
        obj, attributes=["alpha"], extra_attributes={"gamma": "extra"}
    )
    assert "Sample[gamma='extra'](alpha=1)" in with_extra
