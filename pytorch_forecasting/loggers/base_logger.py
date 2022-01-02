from typing import Dict, List

import torch
from pytorch_lightning.loggers.base import LightningLoggerBase
from matplotlib.figure import Figure

from pytorch_forecasting import MultiEmbedding


class ForecastingLoggerBase(LightningLoggerBase):
    def __init__(self, *args, **kwargs):
        super(ForecastingLoggerBase, self).__init__(*args, **kwargs)

    def add_figure(self, caption: str, figure: Figure, step: int) -> None:
        pass

    def add_embeddings(self, input_embeddings: MultiEmbedding, embeddings_labels: Dict[str, List[str]], step: int) -> None:
        pass

    def log_gradient_flow(self, named_parameters: Dict[str, torch.Tensor], step: int) -> None:
        pass
