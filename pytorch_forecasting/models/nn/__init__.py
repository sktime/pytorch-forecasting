from re import S
from typing import Dict

import torch
from torch import embedding, nn

from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.models.nn.rnn import GRU, LSTM, HiddenState, get_rnn
from pytorch_forecasting.utils import TupleOutputMixIn

__all__ = ["MultiEmbedding", "get_rnn", "LSTM", "GRU", "HiddenState", "TupleOutputMixIn"]
