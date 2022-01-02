from typing import List, Dict

import torch
import wandb
from matplotlib.figure import Figure
from pytorch_lightning.loggers import WandbLogger

from pytorch_forecasting import MultiEmbedding
from pytorch_forecasting.loggers.base_logger import ForecastingLoggerBase


class ForecastingWandbLogger(WandbLogger, ForecastingLoggerBase):
    def __init__(self, *args, **kwargs):
        super(ForecastingWandbLogger, self).__init__(*args, **kwargs)

    def add_figure(self, caption: str, figure: Figure, step: int) -> None:
        self.experiment.log({caption: wandb.Image(figure)}, step=step)

    def add_embeddings(
        self, input_embeddings: MultiEmbedding, embeddings_labels: Dict[str, List[str]], step: int
    ) -> None:
        for name, emb in input_embeddings.items():
            labels = embeddings_labels[name]
            self.experiment.log({f"embeddings_{name}": wandb.Table(columns=labels, data=emb)}, step=step)

    def log_gradient_flow(self, named_parameters: Dict[str, torch.Tensor], step: int) -> None:
        pass
