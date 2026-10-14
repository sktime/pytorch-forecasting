from collections import namedtuple

import pytest
import torch

from pytorch_forecasting.utils import TupleOutputMixIn, move_to_device


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_move_to_device_network_output(device):
    prediction = torch.nn.Linear(3, 2)(torch.ones(2, 3))
    output = TupleOutputMixIn().to_network_output(
        prediction=prediction,
        decoder_lengths=torch.tensor([2, 1]),
        targets=[torch.ones(2), torch.zeros(2)],
        optional=None,
    )
    moved = move_to_device(output, device)
    assert type(moved) is type(output)
    assert moved.prediction.device == torch.device(device)
    assert moved.prediction.shape == prediction.shape
    assert moved.prediction.requires_grad == prediction.requires_grad
    assert moved.decoder_lengths.device == torch.device(device)
    assert all(target.device == torch.device(device) for target in moved.targets)
    assert moved.optional is None


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_move_to_device_nested_containers(container, device):
    values = container(
        [{"prediction": torch.ones(2)}, [], container([None, torch.zeros(2)])]
    )
    moved = move_to_device(values, device)
    assert type(moved) is container
    assert type(moved[2]) is container
    assert moved[0]["prediction"].device == torch.device(device)
    assert moved[1] == []
    assert moved[2][0] is None
    assert moved[2][1].device == torch.device(device)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_move_to_device_empty_sequence(container, device):
    moved = move_to_device(container(), device)
    assert type(moved) is container
    assert len(moved) == 0


@pytest.mark.parametrize("container", [list, tuple])
def test_move_to_device_checks_every_tensor(container):
    values = container([torch.empty(2, device="meta"), torch.ones(2)])
    moved = move_to_device(values, "meta")
    assert type(moved) is container
    assert all(value.device == torch.device("meta") for value in moved)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_move_to_device_namedtuple(device):
    pair_type = namedtuple("Pair", ["prediction", "lengths"])
    moved = move_to_device(pair_type(torch.ones(2), torch.tensor([2, 1])), device)
    assert type(moved) is pair_type
    assert moved.prediction.device == torch.device(device)
    assert moved.lengths.device == torch.device(device)
