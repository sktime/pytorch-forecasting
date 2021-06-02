
from torch.nn.utils import weight_norm
from torch.nn import Module
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:
# Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.
# Official TCN PyTorch implementation: https://github.com/locuslab/TCN
from torch import nn

class GAP1d(Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d,self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
    def forward(self, x):
        res=self.gap(x)
        return res.view(res.size(0), -1) ##fastai's layer has problems with latest torch


class Chomp1d(Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock,self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(ni,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

def TemporalConvNet(c_in, layers, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size, padding=(ks-1) * dilation_size, dropout=dropout)]
    return nn.Sequential(*temp_layers)

class Temporal_Convolutional_Networks(Module):
    def __init__(self, c_in, c_out, layers=8*[25], ks=7, conv_dropout=0., fc_dropout=0.):
        super(Temporal_Convolutional_Networks,self).__init__()
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)

from typing import Dict, List, Tuple

from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.models import BaseModelWithCovariates

class TCN(BaseModelWithCovariates):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_hidden_layers: List[int],
        conv_dropout:float,
        fc_dropout:float,
        kernel_size:int,
        x_reals: List[str],
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        embedding_paddings: List[str],
        categorical_groups: Dict[str, List[str]],
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # create embedder - can be fed with x["encoder_cat"] or x["decoder_cat"] and will return
        # dictionary of category names mapped to embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        # calculate the size of all concatenated embeddings + continous variables
        n_features = sum(
            embedding_size for classes_size, embedding_size in self.hparams.embedding_sizes.values()
        ) + len(self.reals)

        # create network that will be fed with continious variables and embeddings
        self.network = Temporal_Convolutional_Networks(
            c_in=n_features,
            c_out=self.hparams.output_size,
            layers=self.hparams.n_hidden_layers,
            ks=self.hparams.kernel_size,
            conv_dropout=self.hparams.conv_dropout,
            fc_dropout=self.hparams.fc_dropout

        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_lengths"].size(0)
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
        prediction = self.network(network_input.permute(0,2,1))

        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"

        return super().from_dataset(dataset, **new_kwargs)

