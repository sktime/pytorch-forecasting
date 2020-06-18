import inspect
from typing import Union, Dict, List, Tuple
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder


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
        categoricals_encoders=None,
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

        self.data = data.sort_values(self.group_ids + [self.time_idx])
        self.data_index = self.get_index_filtering()

        if categoricals_encoders is None:
            self.categoricals_encoders = {name: LabelEncoder().fit(self.data[name]) for name in self.categoricals}
        else:
            self.categoricals_encoders = categoricals_encoders

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

        new = TimeSeriesDataSet(data, **kwargs)
        return new

    def get_index_filtering(self):

        g = self.data.groupby(self.group_ids)

        df_index_first = g[self.time_idx].transform("nth", 0).to_frame("time_first")
        df_index_last = g[self.time_idx].transform("nth", -1).to_frame("time_last")
        df_index_diff_to_next = -g[self.time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
        df_index = pd.concat([df_index_first, df_index_last, df_index_diff_to_next], axis=1)
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = self.data[self.time_idx]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(int) + 1

        # calculate maxium index to include from current index_start
        df_index["index_end"] = df_index["index_start"]
        max_time = (df_index["time"] + self.max_encode_length + self.max_prediction_length).clip(
            upper=df_index["count"]
        )

        for _ in range(df_index["count"].max()):
            new_end_time = df_index[["time", "time_diff_to_next"]].iloc[df_index["index_end"]].sum(axis=1).to_numpy()
            df_index["index_end"] = df_index["index_end"].where(new_end_time + 1 > max_time, df_index["index_end"] + 1,)

        # filter out where encode and decode length are not satisfied
        encoded = df_index["time"].iloc[df_index["index_end"]].to_numpy() - df_index["time"] + 1
        df_index = df_index[encoded == self.max_encode_length + self.max_prediction_length]  # todo: do not filter
        return df_index

    def __len__(self):
        # todo: set to number of potential existing sequences - also last datapoints are not useable
        return self.data_index.shape[0]

    def __getitem__(self, idx):

        data = self.data.iloc[self.data_index.index_start.iloc[idx] : self.data_index.index_end.iloc[idx] + 1]

        # todo: handle missings
        # todo: add indicator about how many encodes
        categoricals = np.stack(
            [self.categoricals_encoders[name].transform(data[name]) for name in self.categoricals], axis=1
        )
        reals = data[self.reals].to_numpy(dtype=np.float)
        target = data[self.target].to_numpy(dtype=np.float)
        return dict(x_cat=categoricals, x_cont=reals), target
