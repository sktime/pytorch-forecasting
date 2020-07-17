"""
Timeseries data is special and has to be processed
"""
from copy import deepcopy
import inspect
from typing import Union, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch
from torch.distributions import Binomial, Beta
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class NaNLabelEncoder(LabelEncoder):
    """
    Labelencoder that can optionally always encode nan as class 0
    """

    def __init__(self, add_nan: bool = False):
        """
        init NaNLabelEncoder

        Args:
            add_nan: if to force encoding of nan at 0
        """
        self.add_nan = add_nan

    def fit_transform(self, y):
        if self.add_nan:
            self.fit(y)
            return self.transform(y)
        return super().transform(y)

    def is_numeric(self, y):
        return y.dtype.kind in "bcif" or (isinstance(y, pd.CategoricalDtype) and y.cat.categories.dtype.kind in "bcif")

    def encode_nans(self, y):
        if not self.is_numeric(y) and isinstance(y, pd.CategoricalDtype):
            if "nan" not in y.cat.categories:
                y = y.cat.add_categories("nan")
            y = y.fillna("nan")
        return y

    def fit(self, y):
        super().fit(y)
        if self.add_nan:
            y = self.encode_nans(y)
            self.classes_ = np.asarray(
                [["nan", np.nan][self.is_numeric(y)]] + [c for c in self.classes_ if c not in [np.nan, "nan"]]
            )
        return self

    def transform(self, y):
        if self.add_nan:
            y = self.encode_nans(y)
        return super().transform(y)


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
        max_encoder_length: int = 30,
        min_encoder_length: int = 0,
        min_prediction_idx: int = None,
        min_prediction_length: int = 1,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        dropout_categoricals: List[str] = [],
        add_relative_time_idx: bool = True,
        constant_fill_strategy={},
        categoricals_encoders={},
        scalers={},
        randomize_length: Union[None, Tuple[float, float]] = (0.2, 0.05),
        predict_mode: bool = False,
    ):
        """
        Timeseries dataset

        Args:
            data: dataframe with sequence data - each row can be identified with ``time_idx`` and the ``group_ids``
            time_idx: integer column denoting the time index
            target: float column denoting the target
            group_ids: list of column names identifying a timeseries
            weight: column name for weights
            max_encoder_length: maximum length to encode
            min_encoder_length: minimum allowed length to encode
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
            dropout_categoricals: list of categorical variables that are unknown when making a forecast without
                observed history
            add_relative_time_idx: if to add a relative time index as feature
            constant_fill_strategy: dictionary of column names with constants to fill in missing values if there are
                gaps in the sequence
                (otherwise forward fill strategy is used)
            categoricals_encoders: dictionary of scikit learn label transformers or None
            scalers: dictionary of scikit learn scalers or None
            randomize_length: None if not to randomize lengths. Tuple of beta distribution concentrations from which
                probabilities are sampled that are used to sample new sequence lengths with a binomial distribution
            predict_mode: if to only iterate over each timeseries once
        """
        super().__init__()
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert self.min_prediction_length > 0, "prediction length must be larger than 0"
        self.target = target
        self.weight = weight
        self.time_idx = time_idx
        self.group_ids = group_ids
        self.static_categoricals = [] + static_categoricals
        self.static_reals = [] + static_reals
        self.time_varying_known_categoricals = [] + time_varying_known_categoricals
        self.time_varying_known_reals = [] + time_varying_known_reals
        self.time_varying_unknown_categoricals = [] + time_varying_unknown_categoricals
        self.time_varying_unknown_reals = [] + time_varying_unknown_reals
        self.dropout_categoricals = [] + dropout_categoricals
        self.add_relative_time_idx = add_relative_time_idx
        self.randomize_length = randomize_length
        self.min_prediction_idx = min_prediction_idx or data[self.time_idx].min()
        self.constant_fill_strategy = constant_fill_strategy
        self.predict_mode = predict_mode

        assert (
            self.target not in self.time_varying_known_reals
        ), "target should be an unknown continuous variable in the future"

        # set data
        assert data.index.is_unique, "data index has to be unique"
        data = data.sort_values(self.group_ids + [self.time_idx])

        # encode categoricals
        self.categoricals_encoders = categoricals_encoders
        for name in set(self.categoricals + self.group_ids):
            if name not in self.categoricals_encoders:
                self.categoricals_encoders[name] = NaNLabelEncoder(add_nan=name in self.dropout_categoricals).fit(
                    data[name]
                )
            if self.categoricals_encoders[name] is not None:
                data[name] = self.categoricals_encoders[name].transform(data[name])

        # scale continuous variables
        self.scalers = scalers
        data["__time_idx__"] = data[self.time_idx]  # save unscaled
        data["__target__"] = data[self.target]
        if self.weight is not None:
            data["__weight__"] = data[self.weight]

        # add time index relative to prediction position
        if self.add_relative_time_idx:
            if "relative_time_idx" not in self.time_varying_known_reals:
                self.time_varying_known_reals.append("relative_time_idx")
            data["relative_time_idx"] = 0.0  # dummy - real value will be set dynamiclly in __getitem__()

        # rescale continuous variables
        for name in self.reals:
            if name not in self.scalers:
                if name == self.time_idx:
                    self.scalers[name] = MinMaxScaler(feature_range=(-1, 1)).fit(data[[name]])
                else:
                    self.scalers[name] = StandardScaler().fit(data[[name]])
            if self.scalers[name] is not None:
                data[name] = self.scalers[name].transform(data[[name]]).reshape(-1)

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
        self.index = self.construct_index(data, predict_mode=predict_mode)

        # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)

    def _data_to_tensors(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:

        index = torch.tensor(data[self.group_ids].to_numpy(np.long))
        time = torch.tensor(data["__time_idx__"].to_numpy(np.long))

        categorical = torch.tensor(data[self.categoricals].to_numpy(np.long))

        if self.weight is not None:
            target_names = ["__target__", "__weight__"]
        else:
            target_names = "__target__"

        target = torch.tensor(data[target_names].to_numpy(dtype=np.float32))
        continuous = torch.tensor(data[self.reals].to_numpy(dtype=np.float32))

        return dict(reals=continuous, categoricals=categorical, groups=index, target=target, time=time)

    @property
    def categoricals(self):
        return self.static_categoricals + self.time_varying_known_categoricals + self.time_varying_unknown_categoricals

    @property
    def reals(self):
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        kwargs = {
            name: getattr(self, name) for name in inspect.signature(self.__class__).parameters.keys() if name != "data"
        }
        kwargs["categoricals_encoders"] = self.categoricals_encoders
        kwargs["scalers"] = self.scalers
        return kwargs

    @classmethod
    def from_dataset(
        cls, dataset, data: pd.DataFrame, stop_randomization: bool = True, predict: bool = True, **update_kwargs
    ):
        return cls.from_parameters(
            dataset.get_parameters(), data, stop_randomization=stop_randomization, predict=predict, **update_kwargs
        )

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        stop_randomization: bool = True,
        predict: bool = True,
        **update_kwargs,
    ):
        parameters = deepcopy(parameters)
        if predict:
            assert stop_randomization, "if predicting, no randomization should be possible"
            parameters["min_prediction_length"] = parameters["max_prediction_length"]
            parameters["predict_mode"] = True
        if stop_randomization:
            parameters["randomize_length"] = None
        parameters.update(update_kwargs)

        new = cls(data, **parameters)
        return new

    def construct_index(self, data: pd.DataFrame, predict_mode: bool) -> pd.DataFrame:

        g = data.groupby(self.group_ids, observed=True)

        df_index_first = g["__time_idx__"].transform("nth", 0).to_frame("time_first")
        df_index_last = g["__time_idx__"].transform("nth", -1).to_frame("time_last")
        df_index_diff_to_next = -g["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
        df_index = pd.concat([df_index_first, df_index_last, df_index_diff_to_next], axis=1)
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = data["__time_idx__"]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(int) + 1
        df_index["group_id"] = g.ngroup()

        # calculate maxium index to include from current index_start
        max_time = (df_index["time"] + self.max_encoder_length + self.max_prediction_length).clip(
            upper=df_index["count"] + df_index.time_first
        )

        # if there are missing timesteps, we cannot say directly what is the last timestep to include
        # therefore we iterate until it is found
        if (df_index["time_diff_to_next"] != 1).any():
            df_index["index_end"] = df_index["index_start"]
            for _ in range(df_index["count"].max()):
                new_end_time = (
                    df_index[["time", "time_diff_to_next"]].iloc[df_index["index_end"]].sum(axis=1).to_numpy()
                )
                df_index["index_end"] = df_index["index_end"].where(
                    new_end_time + 1 > max_time, df_index["index_end"] + 1
                )
        else:
            # direct calculation of end index if there are no missing timesteps in the data
            df_index["index_end"] = df_index["index_start"] + (max_time - df_index["time"] - 1)

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = df_index["time"].iloc[df_index["index_end"]].to_numpy() - df_index["time"] + 1

        if predict_mode:  # keep longest element per series (i.e. the first element that spans to the end of the series)
            # filter all elements that are longer than the allowed maximum sequence length
            df_index = df_index[
                lambda x: (x["time_last"] - x["time"] + 1 <= self.max_prediction_length + self.max_encoder_length)
                & (x["sequence_length"] >= self.min_prediction_length + self.min_encoder_length)
            ]
            # choose longest sequence
            df_index = df_index.loc[df_index.groupby("group_id").sequence_length.idxmax()]
        else:
            df_index = df_index[
                # sequence must be at least of minimal prediction length
                lambda x: (x.sequence_length >= self.min_prediction_length + self.min_encoder_length)
                &
                # prediction must be for after minimal prediction index + length of prediction
                (x["sequence_length"] + x["time"] >= self.min_prediction_idx - 1 + self.min_prediction_length)
            ]

        return df_index

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1]
        data_cat = self.data["categoricals"][index.index_start : index.index_end + 1]
        time = self.data["time"][index.index_start : index.index_end + 1]
        target = self.data["target"][index.index_start : index.index_end + 1]

        sequence_length = len(time)

        # fill in missing values (if not all time indices are specified
        if sequence_length < index.sequence_length:
            repetitions = torch.cat([time[1:] - time[:-1], torch.ones(1, dtype=time.dtype)])
            indices = torch.repeat_interleave(torch.arange(len(time)), repetitions)
            repetition_indices = torch.cat([torch.tensor([False], dtype=torch.bool), indices[1:] == indices[:-1]])

            # select data
            data_cat = data_cat[indices]
            data_cont = data_cont[indices]
            target = target[indices]

            # reset index
            if self.time_idx in self.reals:
                time_idx = self.reals.index(self.time_idx)
                data_cont[:, time_idx] = torch.linspace(
                    data_cont[0, time_idx], data_cont[-1, time_idx], len(target), dtype=data_cont.dtype
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                col_idx = self.reals.index(name)
                data_cont[repetition_indices, col_idx] = value

            sequence_length = len(target)

        # determine data window
        assert sequence_length >= self.min_prediction_length
        # determine prediction/decode length and encode length
        decoder_length = min(time[-1] - (self.min_prediction_idx - 1), self.max_prediction_length, sequence_length)
        encoder_length = sequence_length - decoder_length
        assert decoder_length >= self.min_prediction_length

        if self.randomize_length is not None:  # randomization improves generalization
            # modify encode and decode lengths
            encoder_length_probability, decoder_length_probability = Beta(*self.randomize_length).sample(
                torch.Size([2])
            )

            # subsample a new/smaller encode length
            new_encoder_length = int(Binomial(encoder_length, encoder_length_probability).sample())

            # sample a new/smaller decode length
            new_decoder_length = int(
                max(self.min_prediction_length, Binomial(decoder_length, encoder_length_probability).sample()),
            )
            # select subset of sequence of new sequence
            if new_encoder_length + new_decoder_length < len(target):
                data_cat = data_cat[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                data_cont = data_cont[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                target = target[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                encoder_length = new_encoder_length
                decoder_length = new_decoder_length

            # switch some variables to nan if encode length is 0
            if encoder_length == 0 and len(self.dropout_categoricals) > 0:
                data_cat[:, [self.categoricals.index(c) for c in self.dropout_categoricals]] = 0  # zero is encoded nan

        assert decoder_length > 0
        assert encoder_length >= 0

        if self.add_relative_time_idx:
            data_cont[:, self.reals.index("relative_time_idx")] = (
                torch.arange(-encoder_length, decoder_length, dtype=data_cont.dtype) / self.max_encoder_length
            )

        return (
            dict(
                x_cat=data_cat, x_cont=data_cont, encoder_length=encoder_length, encoder_target=target[:encoder_length],
            ),
            target[encoder_length:],
        )

    def _collate_fn(self, batches):
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([len(batch[1]) for batch in batches], dtype=torch.long)

        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        target = rnn.pad_sequence([batch[1] for batch in batches], batch_first=True)
        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
            ),
            target,
        )

    def to_dataloader(self, train: bool = True, **kwargs):
        return DataLoader(self, shuffle=train, drop_last=train, collate_fn=self._collate_fn, **kwargs)

    def get_index(self) -> pd.DataFrame:
        """
        Index

        Returns:
            dataframe with time index column for first prediction and group ids
        """
        decoder_length = pd.DataFrame(
            dict(
                prediction_idx=self.data["time"][self.index.index_end.to_numpy()] - (self.min_prediction_idx - 1),
                sequence_length=self.index.sequence_length,
                max_prediction_length=self.max_prediction_length,
            )
        ).min(axis=1)
        encoder_lengths = self.index.sequence_length - decoder_length
        index_data = {self.time_idx: self.index.time + encoder_lengths}
        for id in self.group_ids:
            index_data[id] = self.data["groups"][:, self.group_ids.index(id)][self.index.index_start.to_numpy()]
            # decode if possible
            if id in self.categoricals_encoders:
                index_data[id] = self.categoricals_encoders[id].inverse_transform(index_data[id])
        index = pd.DataFrame(index_data, index=self.index.index)
        return index

    @property
    def scales(self) -> Dict[str, Tuple[float, float]]:
        """Mean and scale for each real type column."""
        scales = {}
        for name, scaler in self.scalers.items():
            if isinstance(scaler, MinMaxScaler):
                mean = scaler.data_min_[0] + scaler.data_max_[0] / 2
                scale = scaler.scale_[0]
            elif isinstance(scaler, StandardScaler):
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
            else:
                raise ValueError(f"Scales extraction for scaler of type {type(scaler)} not implemented")
            scales[name] = (mean, scale)
        return scales

