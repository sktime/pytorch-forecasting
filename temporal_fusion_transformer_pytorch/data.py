import inspect
from typing import Union, Dict, List, Tuple
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class TimeSeriesDataSet(Dataset):
    """Dataset Basic Structure for Temporal Fusion Transformer"""

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
        fill_stragegy={},
        categoricals_encoders={},
        scalers={},
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
        self.fill_stragegy = fill_stragegy

        # set data
        self.data = data.sort_values(self.group_ids + [self.time_idx])

        # encode categoricals
        self.categoricals_encoders = categoricals_encoders
        for name in self.categoricals:
            if name not in self.categoricals_encoders:
                self.categoricals_encoders[name] = LabelEncoder().fit(self.data[name])
            self.data[name] = self.categoricals_encoders[name].transform(self.data[name])

        # scale continuous variables
        self.scalers = scalers
        self.data["__time_idx__"] = self.data[self.time_idx]
        for name in self.reals:
            if name not in self.scalers:
                if name == self.time_idx:
                    self.scalers[name] = MinMaxScaler().fit(self.data[[name]])
                else:
                    self.scalers[name] = StandardScaler().fit(self.data[[name]])
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
    def from_dataset(dataset, data):
        kwargs = {
            name: getattr(dataset, name)
            for name in inspect.signature(TimeSeriesDataSet).parameters.keys()
            if name != "data"
        }
        kwargs["categoricals_encoders"] = dataset.categoricals_encoders
        kwargs["scalers"] = dataset.scalers

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
        encoded = df_index["time"].iloc[df_index["index_end"]].to_numpy() - df_index["time"] + 1
        filter = encoded == self.max_encode_length + self.max_prediction_length
        assert filter.sum() > 0, "no samples are remaining after applying the filter"
        df_index = df_index[filter]  # todo: do not filter
        return df_index

    def __len__(self):
        # todo: set to number of potential existing sequences - also last datapoints are not useable
        return self.data_index.shape[0]

    def __getitem__(self, idx):
        data = self.data.iloc[self.data_index.index_start.iloc[idx] : self.data_index.index_end.iloc[idx] + 1]

        # todo: handle missings
        # todo: add indicator about how many encodes
        categoricals = data[self.categoricals].to_numpy(np.long)
        reals = data[self.reals].to_numpy(dtype=np.float32)
        target = data[self.target].to_numpy(dtype=np.float32)[-self.max_prediction_length :]
        return dict(x_cat=categoricals, x_cont=reals), target
