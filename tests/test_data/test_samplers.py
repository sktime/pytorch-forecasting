import pytest
import torch
from torch.utils.data.sampler import SequentialSampler

from pytorch_forecasting.data import TimeSynchronizedBatchSampler


@pytest.mark.parametrize(
    "drop_last,shuffle,as_string,batch_size",
    [
        (True, True, True, 64),
        (False, False, False, 64),
        (True, False, False, 1000),
    ],
)
def test_TimeSynchronizedBatchSampler(
    test_dataset, shuffle, drop_last, as_string, batch_size
):
    if as_string:
        dataloader = test_dataset.to_dataloader(
            batch_sampler="synchronized",
            shuffle=shuffle,
            drop_last=drop_last,
            batch_size=batch_size,
        )
    else:
        sampler = TimeSynchronizedBatchSampler(
            SequentialSampler(test_dataset),
            shuffle=shuffle,
            drop_last=drop_last,
            batch_size=batch_size,
        )
        dataloader = test_dataset.to_dataloader(batch_sampler=sampler)

    time_idx_pos = test_dataset.reals.index("time_idx")
    for x, _ in iter(dataloader):  # check all samples
        time_idx_of_first_prediction = x["decoder_cont"][:, 0, time_idx_pos]
        assert torch.isclose(
            time_idx_of_first_prediction, time_idx_of_first_prediction[0]
        ).all(), "Time index should be the same for the first prediction"
