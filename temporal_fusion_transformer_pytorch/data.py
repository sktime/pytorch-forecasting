import inspect
from typing import Union, Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.distributions import Binomial, Beta
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class TimeSeriesDataSet(Dataset):
    """Dataset Basic Structure for Temporal Fusion Transformer"""

    # todo: automatic skew
    # todo: handle missings
    # TODO: support omissions of variables, e.g. SKU empty
    def __init__(
        self,
        data,
        time_idx: str,
        target: str,
        group_ids: List[str],
        max_encode_length: int,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        add_relative_time_idx: bool = True,
        fill_stragegy={},
        categoricals_encoders={},
        scalers={},
        randomize_length: Union[None, Tuple[float, float]] = (0.2, 0.05),
    ):
        super().__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.max_encode_length = max_encode_length
        self.max_prediction_length = max_prediction_length
        self.target = target
        self.time_idx = time_idx
        self.group_ids = group_ids
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.add_relative_time_idx = add_relative_time_idx
        self.randomize_length = randomize_length
        assert (
            self.target in self.time_varying_unknown_reals
        ), "target should be an unknown continuous variable in the future"
        self.fill_stragegy = fill_stragegy

        # set data
        self.data = data.sort_values(self.group_ids + [self.time_idx])

        # encode categoricals
        self.categoricals_encoders = categoricals_encoders
        for name in self.categoricals:
            if name not in self.categoricals_encoders:
                self.categoricals_encoders[name] = LabelEncoder().fit(self.data[name])
            if self.categoricals_encoders[name] is not None:
                self.data[name] = self.categoricals_encoders[name].transform(self.data[name])

        # scale continuous variables
        self.scalers = scalers
        self.data["__time_idx__"] = self.data[self.time_idx]  # save unscaled
        self.data["__target__"] = self.data[self.target]

        # add time index relative to prediction position
        if self.add_relative_time_idx:
            if "relative_time_idx" not in self.time_varying_known_reals:
                self.time_varying_known_reals.append("relative_time_idx")
            self.data["relative_time_idx"] = 0.0  # dummy - real value will be set dynamiclly in __get_item__()

        # rescale continuous variables
        for name in self.reals:
            if name not in self.scalers:
                if name == self.time_idx:
                    self.scalers[name] = MinMaxScaler().fit(self.data[[name]])
                else:
                    self.scalers[name] = StandardScaler().fit(self.data[[name]])
            if self.scalers[name] is not None:
                self.data[name] = self.scalers[name].transform(self.data[[name]]).reshape(-1)

        # create index
        self.data_index = self.construct_index()

    @property
    def categoricals(self):
        return self.static_categoricals + self.time_varying_known_categoricals + self.time_varying_unknown_categoricals

    @property
    def reals(self):
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals

    @staticmethod
    def from_dataset(dataset, data: pd.DataFrame, stop_randomization: bool = True):
        kwargs = {
            name: getattr(dataset, name)
            for name in inspect.signature(TimeSeriesDataSet).parameters.keys()
            if name != "data"
        }
        kwargs["categoricals_encoders"] = dataset.categoricals_encoders
        kwargs["scalers"] = dataset.scalers
        if stop_randomization:
            kwargs["randomize_length"] = None

        new = TimeSeriesDataSet(data, **kwargs)
        return new

    def construct_index(self):

        g = self.data.groupby(self.group_ids)

        df_index_first = g["__time_idx__"].transform("nth", 0).to_frame("time_first")
        df_index_last = g["__time_idx__"].transform("nth", -1).to_frame("time_last")
        df_index_diff_to_next = -g["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
        df_index = pd.concat([df_index_first, df_index_last, df_index_diff_to_next], axis=1)
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = self.data["__time_idx__"]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(int) + 1

        # calculate maxium index to include from current index_start
        df_index["index_end"] = df_index["index_start"]
        max_time = (df_index["time"] + self.max_encode_length + self.max_prediction_length).clip(
            upper=df_index["count"] + df_index.time_first
        )

        for _ in range(df_index["count"].max()):
            new_end_time = df_index[["time", "time_diff_to_next"]].iloc[df_index["index_end"]].sum(axis=1).to_numpy()
            df_index["index_end"] = df_index["index_end"].where(new_end_time + 1 > max_time, df_index["index_end"] + 1)
        return df_index

    def __len__(self):
        # todo: set to number of potential existing sequences - also last datapoints are not useable
        return self.data_index.shape[0]

    def __getitem__(self, idx):
        # get index data
        data = self.data.iloc[self.data_index.index_start.iloc[idx] : self.data_index.index_end.iloc[idx] + 1].copy()

        # todo: handle missings -> fill them up with strategy
        # determine data window
        sequence_length = len(data)
        max_prediction_length = self.max_prediction_length
        if self.randomize_length is not None:
            # modify sequence length
            sequence_length_prob, encode_length_probability = Beta(*self.randomize_length).sample(torch.Size([2]))
            sequence_length = int(max(1, Binomial(sequence_length, sequence_length_prob).sample()))
            max_prediction_length = int(max(1, Binomial(max_prediction_length, encode_length_probability).sample()))
            if sequence_length < len(data):
                data = data.iloc[-sequence_length:]  # select subset of sequence

        encode_length = min(max(0, sequence_length - max_prediction_length), self.max_encode_length)
        decode_length = sequence_length - encode_length

        # extract data
        target = data["__target__"].to_numpy(dtype=np.float32)
        if self.add_relative_time_idx:
            data["relative_time_idx"] = np.arange(-encode_length, decode_length, dtype=float) / self.max_encode_length
        categoricals = data[self.categoricals].to_numpy(np.long)
        reals = data[self.reals + ["__target__"]].to_numpy(dtype=np.float32)

        return dict(x_cat=categoricals, x_cont=reals, encode_length=encode_length), target[encode_length:]

    def _collate_fn(self, batches):
        encode_lengths = torch.LongTensor([batch[0]["encode_length"] for batch in batches])
        decode_lengths = torch.LongTensor([len(batch[1]) for batch in batches])

        encoder_cont = rnn.pad_sequence(
            [torch.Tensor(batch[0]["x_cont"][:length]) for length, batch in zip(encode_lengths, batches)],
            batch_first=True,
        )
        encoder_cat = rnn.pad_sequence(
            [torch.LongTensor(batch[0]["x_cat"][:length]) for length, batch in zip(encode_lengths, batches)],
            batch_first=True,
        )
        decoder_cont = rnn.pad_sequence(
            [torch.Tensor(batch[0]["x_cont"][length:]) for length, batch in zip(encode_lengths, batches)],
            batch_first=True,
        )
        decoder_cat = rnn.pad_sequence(
            [torch.LongTensor(batch[0]["x_cat"][length:]) for length, batch in zip(encode_lengths, batches)],
            batch_first=True,
        )

        target = rnn.pack_sequence([torch.Tensor(batch[1]) for batch in batches], enforce_sorted=False)
        x_cat = torch.cat((encoder_cat, decoder_cat), dim=1)
        x_cont = torch.cat((encoder_cont, decoder_cont), dim=1)
        return (
            dict(x_cat=x_cat, x_cont=x_cont, encode_lengths=encode_lengths, decode_lengths=decode_lengths),
            target,
        )

    def to_dataloader(self, train: bool = True, **kwargs):
        return DataLoader(self, shuffle=train, drop_last=train, collate_fn=self._collate_fn, **kwargs)
