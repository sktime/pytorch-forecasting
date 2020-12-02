"""
`DeepAR: Probabilistic forecasting with autoregressive recurrent networks
<https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_
which is the one of the most popular forecasting algorithms and is often used as a baseline
"""
from copy import copy
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot_date
import numpy as np
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.distributions as dists
import torch.nn as nn
from torch.nn.utils import rnn
from torch.utils.data.dataloader import DataLoader

from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, DistributionLoss, Metric, NormalDistributionLoss
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.deepar.sub_modules import HiddenState, get_cell
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding


class DeepAR(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: str = None,
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        DeepAR Network.

        The code is based on the article `DeepAR: Probabilistic forecasting with autoregressive recurrent networks
        <https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_.

        Args:
            cell_type (str, optional): Recurrent cell type ["LSTM", "GRU"]. Defaults to "LSTM".
            hidden_size (int, optional): hidden recurrent size - the most important hyperparameter along with
                ``rnn_layers``. Defaults to 10.
            rnn_layers (int, optional): Number of RNN layers - important hyperparameter. Defaults to 2.
            dropout (float, optional): Dropout in RNN layers. Defaults to 0.1.
            static_categoricals: integer of positions of static categorical variables
            static_reals: integer of positions of static continuous variables
            time_varying_categoricals_encoder: integer of positions of categorical variables for encoder
            time_varying_categoricals_decoder: integer of positions of categorical variables for decoder
            time_varying_reals_encoder: integer of positions of continuous variables for encoder
            time_varying_reals_decoder: integer of positions of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            n_validation_samples (int, optional): Number of samples to use for calculating validation metrics.
                Defaults to None, i.e. no sampling at validation stage and using "mean" of distribution for logging
                metrics calculation.
            n_plotting_samples (int, optional): Number of samples to generate for plotting predictions
                during training. Defaults to ``n_validation_samples`` if not None or 100 otherwise.
            target (str, optional): Target variable. Defaults to None.
            loss (DistributionLoss, optional): Distribution loss function. Keep in mind that each distribution
                loss function might have specific requirements for target normalization.
                Defaults to :py:class:`~pytorch_forecasting.metrics.NormalDistributionLoss`.
            logging_metrics (nn.ModuleList, optional): Metrics to log during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]).
        """
        if loss is None:
            loss = NormalDistributionLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if n_plotting_samples is None:
            if n_validation_samples is None:
                n_plotting_samples = n_validation_samples
            else:
                n_plotting_samples = 100
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        assert set(self.encoder_variables) - {target} == set(
            self.decoder_variables
        ), "Encoder and decoder variables have to be the same apart from target variable"
        assert target in time_varying_reals_encoder, "target has to be real"  # todo: remove this restriction

        time_series_rnn = get_cell(cell_type)
        self.rnn = time_series_rnn(
            input_size=self.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
        )

        # add linear layers for argument projects
        self.distribution_projector = nn.Linear(self.hparams.hidden_size, len(self.loss.distribution_arguments))

    @property
    def input_size(self):
        """Input vector size: length of embeddings and real-values variables."""
        cont_size = len(self.reals)
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        return cont_size + cat_size

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            DeepAR network
        """
        # assert fixed encoder and decoder length for the moment
        new_kwargs = {}
        new_kwargs.update(kwargs)
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Create input vector into RNN network

        Args:
            one_off_target: tensor to insert into first position of target. If None (default), remove first time step.
        """
        # create input vector
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_position] = torch.roll(input_vector[..., self.target_position], shifts=1, dims=1)

        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_position] = one_off_target
        else:
            input_vector = input_vector[:, 1:]

        # shift target
        return input_vector

    @property
    def target_position(self):
        return self.hparams.x_reals.index(self.hparams.target)

    def encode(self, x: Dict[str, torch.Tensor]) -> HiddenState:
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        max_encoder_length = x["encoder_lengths"].max()
        assert x["encoder_lengths"].min() > 0
        if max_encoder_length > 1:
            encoder_lengths = x["encoder_lengths"] - 1
            rnn_encoder_lengths = encoder_lengths.where(encoder_lengths > 0, torch.ones_like(encoder_lengths))
            input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
            _, hidden_state = self.rnn(
                rnn.pack_padded_sequence(
                    input_vector, rnn_encoder_lengths.cpu(), enforce_sorted=False, batch_first=True
                )
            )  # second ouput is not needed (hidden state)
            # replace hidden cell with initial input if encoder_length is zero to determine correct initial state
            no_encoding = (encoder_lengths == 0)[None, :, None]  # shape: n_lstm_layers x batch_size x hidden_size
            hidden_state = self.rnn.handle_no_encoding(hidden_state, no_encoding)
        else:
            hidden_state = self.rnn.init_hidden_state(x, self.hparam.hidden_size)
        return hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            input_vector = rnn.pack_padded_sequence(
                input_vector, decoder_lengths.cpu(), enforce_sorted=False, batch_first=True
            )
            decoder_output, _ = self.rnn(
                input_vector,
                hidden_state,
            )
            decoder_output, _ = rnn.pad_packed_sequence(decoder_output, batch_first=True)
            output = self.distribution_projector(decoder_output)
            output_type = "parameters"

        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_position
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            target_scale = target_scale.repeat_interleave(n_samples, 0)

            # make predictions which are fed into next step
            input_target = input_vector[:, 0, target_pos]
            output = []
            for idx in range(input_vector.size(1)):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = input_target
                decoder_output, hidden_state = self.rnn(x, hidden_state)
                normalized_prediction_parameters = self.distribution_projector(decoder_output)
                # transform into real space
                prediction_parameters = self.transform_output(
                    dict(
                        prediction=normalized_prediction_parameters,
                        target_scale=target_scale,
                        prediction_type="parameters",
                    )
                )
                # sample value(s) from distribution
                prediction = self.loss.sample_n(prediction_parameters, 1)[0]  # select first sample
                # normalize prediction prediction
                # todo: how to handle lags (-> need list of lags and positions
                #   -> then if prediction lenght larger than lag start imputing ->
                #   before that let timeseriesdataset take care)?
                normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
                input_target = normalized_prediction[:, 0]  # set next input target to normalized prediction

                # set output to unnormalized samples
                output.append(prediction[:, 0].view(-1, n_samples))  # append as n_batch_samples x n_random_samples
            output = torch.stack(output, dim=1)  # samples at > 0
            output_type = "samples"
        return output, output_type

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        hidden_state = self.encode(x)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"], x["decoder_cont"], one_off_target=x["encoder_cont"][:, -1, self.target_position]
        )

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output, output_type = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
            n_samples=n_samples,
        )
        # return relevant part
        return dict(
            prediction=output,
            prediction_type=output_type,
            groups=x["groups"],
            decoder_time_idx=x["decoder_time_idx"],
            target_scale=x["target_scale"],
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, _ = self.step(x, y, batch_idx, label="val", n_samples=self.hparams.n_validation_samples)  # log loss
        self.log("val_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return log

    def _log_metrics(
        self, x: Dict[str, torch.Tensor], y: torch.Tensor, out: Dict[str, torch.Tensor], label: str = "train"
    ) -> None:

        if out["prediction_type"] == "parameters":
            # use distribution properties to create point prediction
            out = copy(out)  # copy to avoid side-effects but do not deep copy to re-use references
            y_hat_detached = out["prediction"].detach()
            y_hat_point_detached = self.loss.map_x_to_distribution(y_hat_detached).mean.unsqueeze(-1)
            out["prediction"] = y_hat_point_detached
            out["prediction_type"] = "samples"
        super()._log_metrics(x, y, out, label=label)

    def _log_prediction(self, x, out, batch_idx, label) -> None:
        log_interval = self.log_interval(label == "train")
        if (
            out["prediction_type"] == "parameters"
            and (batch_idx % log_interval == 0 or log_interval < 1.0)
            and log_interval > 0
        ):
            out = copy(out)  # copy to avoid side-effects but do not deep copy to re-use references
            # sample from distribution to create valid prediction
            y_hat_detached = out["prediction"].detach()
            if self.hparams.n_plotting_samples is None:
                y_hat_samples = self.loss.map_x_to_distribution(y_hat_detached).mean.unsqueeze(-1)
            else:
                y_hat_samples = self.loss.sample_n(y_hat_detached, self.hparams.n_plotting_samples).permute(1, 2, 0)
            out["prediction"] = y_hat_samples
            out["prediction_type"] = "samples"
        super()._log_prediction(x, out, batch_idx=batch_idx, label=label)

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
    ) -> plt.Figure:
        # workaround for not being able to compute loss for single sample without parameters of distribution
        return super().plot_prediction(
            x, out, idx=idx, add_loss_to_title=False, show_future_observed=show_future_observed, ax=ax
        )

    def transform_output(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        elif self.output_transformer is None:
            out = out["prediction"]
        else:
            # depending on output, transform differently
            if out["prediction_type"] == "samples":  # samples are already rescaled
                out = out["prediction"]
            elif out["prediction_type"] == "parameters":  # parameters need to be rescaled
                out = self.loss.rescale_parameters(
                    out["prediction"], target_scale=out["target_scale"], encoder=self.output_transformer
                )
            else:
                raise ValueError(f"Unknown output type {out['prediction_type']}")
        return out

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        n_samples: int = 100,
    ):
        """
        predict dataloader

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles" or "raw", or tuple ``("raw", output_name)`` where output_name is
                a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index
            return_decoder_lengths: if to return decoder_lengths
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs
            n_samples: number of samples to draw. Defaults to 100.

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        return super().predict(
            data=data,
            mode=mode,
            return_decoder_lengths=return_decoder_lengths,
            return_index=return_index,
            n_samples=n_samples,  # new keyword that is passed to forward function
            return_x=return_x,
            show_progress_bar=show_progress_bar,
            fast_dev_run=fast_dev_run,
            num_workers=num_workers,
            batch_size=batch_size,
        )
