"""
Timeseries data is special and has to be processed
"""
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
    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        target: str,
        group_ids: List[str],
        weight: Union[str, None] = None,
        max_encode_length: int = 30,
        min_prediction_idx: int = None,
        min_prediction_length: int = 1,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        add_relative_time_idx: bool = True,
        constant_fill_strategy={},
        categoricals_encoders={},
        scalers={},
        randomize_length: Union[None, Tuple[float, float]] = (0.2, 0.05),
    ):
        """
        Timeseries dataset

        Args:
            data: dataframe with sequence data - each row can be identified with ``time_idx`` and the ``group_ids``
            time_idx: integer column denoting the time index
            target: float column denoting the target
            group_ids: list of column names identifying a timeseries
            weight: column name for weights
            max_encode_length: maximum length to encode
            min_prediction_idx: minimum time index from where to start predictions
            min_prediction_length: minimum prediction length
            max_prediction_length: maximum prediction length (choose this not too short as it can help convergence)
            static_categoricals: list of categorical variables that do not change over time
            static_reals: list of continuous variables that do not change over time
            time_varying_known_categoricals: list of categorical variables that change over
                time and are know in the future
            time_varying_known_reals: list of continuous variables that change over
                time and are know in the future
            time_varying_unknown_categoricals: list of categorical variables that change over
                time and are not know in the future
            time_varying_unknown_reals: list of continuous variables that change over
                time and are not know in the future
            add_relative_time_idx: if to add a relative time index as feature
            constant_fill_strategy: dictionary of column names with constants to fill in missing values if there are
                gaps in the sequence
                (otherwise forward fill strategy is used)
            categoricals_encoders: dictionary of scikit learn label transformers or None
            scalers: dictionary of scikit learn scalers or None
            randomize_length: None if not to randomize lengths. Tuple of beta distribution concentrations from which
                probabilities are sampled that are used to sample new sequence lengths with a binomial distribution
        """
        super().__init__()
        self.max_encode_length = max_encode_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert self.min_prediction_length > 0, "prediction length must be larger than 0"
        self.target = target
        self.weight = weight
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
        self.min_prediction_idx = min_prediction_idx or data[self.time_idx].min()
        self.constant_fill_strategy = constant_fill_strategy

        assert (
            self.target in self.time_varying_unknown_reals
        ), "target should be an unknown continuous variable in the future"

        # set data
        assert data.index.is_unique, "data index has to be unique"
        self.data = data[
            lambda x: (x[self.time_idx] >= self.min_prediction_idx - self.max_encode_length)  # limit data
        ].sort_values(self.group_ids + [self.time_idx])

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
        if self.weight is not None:
            self.data["__weight__"] = self.data[self.weight]

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

        # encode constant values
        self.encoded_constant_fill_strategy = {}
        for name, value in self.constant_fill_strategy.items():
            if name in self.scalers:
                self.encoded_constant_fill_strategy[name] = self.scalers[name].transform(np.array([[value]]))[0, 0]
            elif name in self.categoricals_encoders:
                self.encoded_constant_fill_strategy[name] = self.categoricals_encoders[name].transform([value])[0]
            else:
                self.encoded_constant_fill_strategy[name] = value

        # create index
        self.data_index = self.construct_index()

    @property
    def categoricals(self):
        return self.static_categoricals + self.time_varying_known_categoricals + self.time_varying_unknown_categoricals

    @property
    def reals(self):
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals

    @staticmethod
    def from_dataset(dataset, data: pd.DataFrame, stop_randomization: bool = True, **update_kwargs):
        kwargs = {
            name: getattr(dataset, name)
            for name in inspect.signature(TimeSeriesDataSet).parameters.keys()
            if name != "data"
        }
        kwargs["categoricals_encoders"] = dataset.categoricals_encoders
        kwargs["scalers"] = dataset.scalers
        if stop_randomization:
            kwargs["randomize_length"] = None
        kwargs.update(update_kwargs)

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

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = df_index["time"].iloc[df_index["index_end"]].to_numpy() - df_index["time"] + 1
        df_index = df_index[
            # sequence must be at least of minimal prediction length
            lambda x: (x.sequence_length > self.min_prediction_length)
            &
            # prediction must be for after minimal prediction index + length of prediction
            (new_end_time >= self.min_prediction_idx - 1 + self.min_prediction_length)
        ]
        return df_index

    def __len__(self):
        # todo: set to number of potential existing sequences - also last datapoints are not useable
        return self.data_index.shape[0]

    def __getitem__(self, idx):
        index = self.data_index.iloc[idx]
        # get index data
        data = self.data.iloc[index.index_start : index.index_end + 1].copy()

        sequence_length = len(data)

        # fill in missing values (if not all time indices are specified
        if sequence_length < index.sequence_length:
            repetitions = -data.__time_idx__.diff(-1).fillna(-1)
            indices = np.repeat(np.arange(len(data)), repetitions)
            repetition_indices = np.where(np.diff(indices, prepend=[-1]) == 0)[0]
            data = data.iloc[indices]
            # make replacements
            for name, value in self.encoded_constant_fill_strategy.items():
                col_idx = data.columns.get_loc(name)
                data.iloc[repetition_indices, col_idx] = value

        # determine data window
        assert sequence_length >= self.min_prediction_length
        # determine prediction/decode length and encode length
        decode_length = min(
            data.iloc[-1]["__time_idx__"] - (self.min_prediction_idx - 1), self.max_prediction_length, sequence_length
        )
        encode_length = sequence_length - decode_length
        assert decode_length >= self.min_prediction_length

        if self.randomize_length is not None:  # randomization improves generalization
            # modify encode and decode lengths
            encode_length_probability, decode_length_probability = Beta(*self.randomize_length).sample(torch.Size([2]))

            # subsample a new/smaller encode length
            new_encode_length = int(Binomial(encode_length, encode_length_probability).sample())

            # sample a new/smaller decode length
            new_decode_length = int(
                max(self.min_prediction_length, Binomial(decode_length, encode_length_probability).sample()),
            )
            # select subset of sequence of new sequence
            if new_encode_length + new_decode_length < len(data):
                data = data.iloc[encode_length - new_encode_length : encode_length + new_decode_length]
                encode_length = new_encode_length
                decode_length = new_decode_length

        assert decode_length > 0
        assert encode_length >= 0
        assert data.iloc[-1]["__time_idx__"] - self.min_prediction_idx + 1 >= decode_length

        # extract data
        if self.weight is None:
            target = data["__target__"].to_numpy(dtype=np.float32)
        else:
            target = data[["__target__", "__weight__"]].to_numpy(dtype=np.float32)
        if self.add_relative_time_idx:
            data.loc[:, "relative_time_idx"] = (
                np.arange(-encode_length, decode_length, dtype=float) / self.max_encode_length
            )
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

    def get_index(self) -> pd.DataFrame:
        """
        calculate index

        Returns:
            dataframe with time index column for first prediction and group ids
        """
        decode_length = pd.DataFrame(
            dict(
                prediction_idx=self.data["__time_idx__"].iloc[self.data_index.index_end].to_numpy()
                - (self.min_prediction_idx - 1),
                sequence_length=self.data_index.sequence_length,
                max_prediction_length=self.max_prediction_length,
            )
        ).min(axis=1)
        encode_lengths = self.data_index.sequence_length - decode_length
        index_data = {self.time_idx: self.data_index.time + encode_lengths}
        for id in self.group_ids:
            index_data[id] = self.data[id].iloc[self.data_index.index_start]
            # decode if possible
            if id in self.categoricals_encoders:
                index_data[id] = self.categoricals_encoders[id].inverse_transform(index_data[id])
        index = pd.DataFrame(index_data, index=self.data_index.index)
        return index
