from typing import Dict, List

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import MultiEmbedding
from pytorch_forecasting.loggers.base_logger import ForecastingLoggerBase


class ForecastingTensorBoardLogger(TensorBoardLogger, ForecastingLoggerBase):
    def __init__(self, *args, **kwargs):
        super(ForecastingTensorBoardLogger, self).__init__(*args, **kwargs)

    def add_figure(self, caption: str, figure: Figure, step: int) -> None:
        self.experiment.add_figure(caption, figure, step)

    def add_embeddings(
        self, input_embeddings: MultiEmbedding, embedding_labels: Dict[str, List[str]], step: int
    ) -> None:
        for name, emb in input_embeddings.items():
            labels = embedding_labels[name]
            self.experiment.add_embedding(emb.weight.data.detach().cpu(), metadata=labels, tag=name, global_step=step)

    def log_gradient_flow(self, named_parameters: Dict[str, torch.Tensor], step: int) -> None:
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is not None and p.requires_grad and "bias" not in name:
                layers.append(name)
                ave_grads.append(p.grad.abs().mean().cpu())
                self.experiment.add_histogram(tag=name, values=p.grad.cpu(), global_step=step)
        fig, ax = plt.subplots()
        ax.plot(ave_grads)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average gradient")
        ax.set_yscale("log")
        ax.set_title("Gradient flow")
        self.add_figure("Gradient flow", fig, step=step)
