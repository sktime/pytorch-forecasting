"""
Temporal Convolutional Network (TCN): An Empirical Evaluation of Generic Convolutional and Recurrent Networks for
Sequence Modeling <https://arxiv.org/abs/1803.01271>`_. TCNs have often a better performance than LSTMs in
time series forecasting while training much faster.
"""

from copy import copy
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models import BaseModelWithCovariates
from pytorch_forecasting.models.nn import MultiEmbedding


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"

    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        res = self.gap(x)
        return res.view(res.size(0), -1)  # fastai's layer has problems with latest torch


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class Flatten(nn.Module):
    def __init__(
        self,
    ):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)  # 保持时序的长度不变，但是内部的特征会发生改变
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalConvolutionalNetwork(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_layer_sizes: List[int] = [128, 128],
        conv_dropout: float = 0.1,
        fc_dropout: float = 0.1,
        kernel_size: int = 16,
        loss: MultiHorizonMetric = None,
        prediction_length: int = 1,
        output_size: Union[int, List[int]] = 1,
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_labels: Dict[str, List[str]] = {},
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_paddings: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Initialize Temporal Convolutional Network Model - use its :py:meth:`~from_dataset` method if possible.

        Based on the article
        `Temporal Convolutional Network (TCN): An Empirical Evaluation of Generic Convolutional and Recurrent Networks
        for Sequence Modeling
        <https://arxiv.org/abs/1803.01271>`_. TCNs have often a better performance than LSTMs in time series forecasting
        while training much faster.

        Args:
            hidden_layer_sizes (int): the number of temporal blocks. In this implementation every temporal block
                has fixe three casual dialation cnn kernel according to the original paper
            conv_dropout (float): dropout rate for the output of every temporal block
            fc_dropout (float): In this implementation, we use pooling and flatten(Gap1D) to make the output into 1D
                dimension and use a linear layer to complete time series forecasting tasks
                fc_drouput is the droup out rate for the output of Gap1D
            kernel_size (int): the kernel size of the cnn
            loss: loss function taking prediction and targets. Defaults to SMAPE.
            prediction_length (int): Length of the prediction. Also known as 'horizon'.
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            loss = SMAPE()
        self.save_hyperparameters()
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # create embedder - can be fed with x["encoder_cat"] or x["decoder_cat"] and will return
        # dictionary of category names mapped to embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        # calculate the size of all concatenated embeddings + continous variables
        n_features = sum(embedding_size for _, embedding_size in self.hparams.embedding_sizes.values()) + len(
            self.reals
        )

        # create network that will be fed with continious variables and embeddings
        self.network = nn.Sequential(
            TemporalConvNet(
                num_inputs=n_features,
                num_channels=self.hparams.hidden_layer_sizes,
                kernel_size=self.hparams.kernel_size,
                dropout=self.hparams.conv_dropout,
            ),
            GAP1d(),
            nn.Dropout(self.hparams.fc_dropout),
        )

        # final layer
        last_layer_size = self.hparams.hidden_layer_sizes[-1]
        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [
                    nn.Linear(last_layer_size, output_size * self.hparams.prediction_length)
                    for output_size in self.hparams.output_size
                ]
            )
        else:
            self.output_layer = nn.Linear(last_layer_size, self.hparams.output_size * self.hparams.prediction_length)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        embeddings = self.input_embeddings(x["encoder_cat"])  # returns dictionary with embedding tensors
        network_input = torch.cat(
            [x["encoder_cont"]]
            + [
                emb
                for name, emb in embeddings.items()
                if name in self.encoder_variables or name in self.static_variables
            ],
            dim=-1,
        )
        output = self.network(network_input.permute(0, 2, 1))

        if self.n_targets > 1:  # if to use multi-target architecture
            output = [
                output_layer(output).view(-1, self.hparams.prediction_length, self.hparams.output_size)
                for output_layer in self.output_layer
            ]
        else:
            output = self.output_layer(output).view(-1, self.hparams.prediction_length, self.hparams.output_size)
        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        output = self.transform_output(output, target_scale=x["target_scale"])
        return self.to_network_output(prediction=output)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = copy(kwargs)
        new_kwargs["prediction_length"] = dataset.max_prediction_length

        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MAE()))
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"

        return super().from_dataset(dataset, **new_kwargs)
