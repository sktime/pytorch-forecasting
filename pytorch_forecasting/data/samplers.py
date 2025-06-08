"""
Samplers for sampling time series from the :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`
"""  # noqa: E501

import warnings

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data.sampler import Sampler


class GroupedSampler(Sampler):
    """
    Samples mini-batches randomly but in a grouped manner.

    This means that the items from the different groups are always sampled together.
    This is an abstract class. Implement the :py:meth:`~get_groups` method which creates groups to be sampled from.
    """  # noqa: E501

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int = 64,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """
        Initialize.

        Args:
            sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            drop_last (bool): if to drop last mini-batch from a group if it is smaller than batch_size.
                Defaults to False.
            shuffle (bool): if to shuffle dataset. Defaults to False.
            batch_size (int, optional): Number of samples in a mini-batch. This is rather the maximum number
                of samples. Because mini-batches are grouped by prediction time, chances are that there
                are multiple where batch size will be smaller than the maximum. Defaults to 64.
        """  # noqa: E501
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                f"but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        # make groups and construct new index to sample from
        groups = self.get_groups(self.sampler)
        self.construct_batch_groups(groups)

    def get_groups(self, sampler: Sampler):
        """
        Create the groups which can be sampled.

        Args:
            sampler (Sampler): will have attribute data_source which is of type TimeSeriesDataSet.

        Returns:
            dict-like: dictionary-like object with data_source.index as values and group names as keys
        """  # noqa: E501
        raise NotImplementedError()

    def construct_batch_groups(self, groups):
        """
        Construct index of batches from which can be sampled
        """
        self._groups = groups
        # calculate sizes of groups
        self._group_sizes = {}
        warns = []
        for name, group in self._groups.items():  # iterate over groups
            if self.drop_last:
                self._group_sizes[name] = len(group) // self.batch_size
            else:
                self._group_sizes[name] = (
                    len(group) + self.batch_size - 1
                ) // self.batch_size
            if self._group_sizes[name] == 0:
                self._group_sizes[name] = 1
                warns.append(name)
        if len(warns) > 0:
            warnings.warn(
                f"Less than {self.batch_size} samples available for "
                f"{len(warns)} prediction times. "
                f"Use batch size smaller than {self.batch_size}. "
                f"First 10 prediction times with small batch sizes: {warns[:10]}"
            )
        # create index from which can be sampled: index is equal to number of batches
        # associate index with prediction time
        self._group_index = np.repeat(
            list(self._group_sizes.keys()), list(self._group_sizes.values())
        )
        # associate index with batch within prediction time group
        self._sub_group_index = np.concatenate(
            [np.arange(size) for size in self._group_sizes.values()]
        )

    def __iter__(self):
        if self.shuffle:  # shuffle samples
            groups = {name: shuffle(group) for name, group in self._groups.items()}
            batch_samples = np.random.permutation(len(self))
        else:
            groups = self._groups
            batch_samples = np.arange(len(self))

        for idx in batch_samples:
            name = self._group_index[idx]
            sub_group = self._sub_group_index[idx]
            sub_group_start = sub_group * self.batch_size
            sub_group_end = sub_group_start + self.batch_size
            batch = groups[name][sub_group_start:sub_group_end]
            yield batch

    def __len__(self):
        return len(self._group_index)


class TimeSynchronizedBatchSampler(GroupedSampler):
    """
    Samples mini-batches randomly but in a time-synchronised manner.

    Time-synchornisation means that the time index of the first decoder samples are aligned across the batch.
    This sampler does not support missing values in the dataset.
    """  # noqa: E501

    def get_groups(self, sampler: Sampler):
        data_source = sampler.data_source
        index = data_source.index
        # get groups, i.e. group all samples by first predict time
        last_time = data_source.data["time"][index["index_end"].to_numpy()].numpy()
        decoder_lengths = data_source.calculate_decoder_length(
            last_time, index.sequence_length
        )
        first_prediction_time = index.time + index.sequence_length - decoder_lengths + 1
        groups = pd.RangeIndex(0, len(index.index)).groupby(first_prediction_time)
        return groups
