"""
`DeepAR: Probabilistic forecasting with autoregressive recurrent networks
<https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_
which is the one of the most popular forecasting algorithms and is often used as a baseline
"""  # noqa: E501

from copy import deepcopy
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from pytorch_forecasting.data.encoders import MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    DistributionLoss,
    MultiLoss,
    MultivariateDistributionLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.models.base import (
    AutoRegressiveBaseModelWithCovariates,
    Prediction,
)
from pytorch_forecasting.models.nn import HiddenState, MultiEmbedding, get_rnn
from pytorch_forecasting.utils import apply_to_list, to_list


class DeepAR(AutoRegressiveBaseModelWithCovariates):
    """DeepAR: Probabilistic forecasting with autoregressive recurrent networks."""

    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_categoricals_encoder: Optional[list[str]] = None,
        time_varying_categoricals_decoder: Optional[list[str]] = None,
        categorical_groups: Optional[dict[str, list[str]]] = None,
        time_varying_reals_encoder: Optional[list[str]] = None,
        time_varying_reals_decoder: Optional[list[str]] = None,
        embedding_sizes: Optional[dict[str, tuple[int, int]]] = None,
        embedding_paddings: Optional[list[str]] = None,
        embedding_labels: Optional[dict[str, np.ndarray]] = None,
        x_reals: Optional[list[str]] = None,
        x_categoricals: Optional[list[str]] = None,
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: Union[str, list[str]] = None,
        target_lags: Optional[dict[str, list[int]]] = None,
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        DeepAR Network.

        The code is based on the article `DeepAR: Probabilistic forecasting with autoregressive recurrent networks
        <https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_.

        By using a Multivariate Loss such as the
        :py:class:`~pytorch_forecasting.metrics.MultivariateNormalDistributionLoss`,
        the network is converted into a `DeepVAR network <http://arxiv.org/abs/1910.03002>`_.

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
            target (str, optional): Target variable or list of target variables. Defaults to None.
            target_lags (Dict[str, Dict[str, int]]): dictionary of target names mapped to list of time steps by
                which the variable should be lagged.
                Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
                add at least the target variables with the corresponding lags to improve performance.
                Defaults to no lags, i.e. an empty dictionary.
            loss (DistributionLoss, optional): Distribution loss function. Keep in mind that each distribution
                loss function might have specific requirements for target normalization.
                Defaults to :py:class:`~pytorch_forecasting.metrics.NormalDistributionLoss`.
            logging_metrics (nn.ModuleList, optional): Metrics to log during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]).
        """  # noqa: E501
        if loss is None:
            loss = NormalDistributionLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if n_plotting_samples is None:
            if n_validation_samples is None:
                n_plotting_samples = n_validation_samples
            else:
                n_plotting_samples = 100
        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if embedding_sizes is None:
            embedding_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if target_lags is None:
            target_lags = {}
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        lagged_target_names = [l for lags in target_lags.values() for l in lags]
        assert set(self.encoder_variables) - set(to_list(target)) - set(
            lagged_target_names
        ) == set(self.decoder_variables) - set(lagged_target_names), (
            "Encoder and decoder variables have to be"
            " the same apart from target variable"
        )
        for targeti in to_list(target):
            assert (
                targeti in time_varying_reals_encoder
            ), f"target {targeti} has to be real"  # todo: remove this restriction
        assert (isinstance(target, str) and isinstance(loss, DistributionLoss)) or (
            isinstance(target, (list, tuple))
            and isinstance(loss, MultiLoss)
            and len(loss) == len(target)
        ), "number of targets should be equivalent to number of loss metrics"

        rnn_class = get_rnn(cell_type)
        cont_size = len(self.reals)
        cat_size = sum(self.embeddings.output_size.values())
        input_size = cont_size + cat_size
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
        )

        # add linear layers for argument projects
        if isinstance(target, str):  # single target
            self.distribution_projector = nn.Linear(
                self.hparams.hidden_size, len(self.loss.distribution_arguments)
            )
        else:  # multi target
            self.distribution_projector = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, len(args))
                    for args in self.loss.distribution_arguments
                ]
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
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
        """  # noqa: E501
        new_kwargs = {}
        if dataset.multi_target:
            new_kwargs.setdefault(
                "loss",
                MultiLoss([NormalDistributionLoss()] * len(dataset.target_names)),
            )
        new_kwargs.update(kwargs)
        assert (
            not isinstance(dataset.target_normalizer, NaNLabelEncoder)
            and (
                not isinstance(dataset.target_normalizer, MultiNormalizer)
                or all(
                    not isinstance(normalizer, NaNLabelEncoder)
                    for normalizer in dataset.target_normalizer
                )
            )
        ), (
            "target(s) should be continuous - categorical targets are not supported"
        )  # todo: remove this restriction # noqa: E501
        if isinstance(new_kwargs.get("loss", None), MultivariateDistributionLoss):
            assert (
                dataset.min_prediction_length == dataset.max_prediction_length
            ), "Multivariate models require constant prediction lenghts"
        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    def construct_input_vector(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        one_off_target: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Create input vector into RNN network

        Args:
            one_off_target: tensor to insert into first position of target.
                If None (default), remove first time step.
        """
        # create input vector
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat(list(embeddings.values()), dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont.clone()

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )

        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]

        # shift target
        return input_vector

    def encode(self, x: dict[str, torch.Tensor]) -> HiddenState:
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        assert x["encoder_lengths"].min() > 0
        encoder_lengths = x["encoder_lengths"] - 1
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        _, hidden_state = self.rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(
            x, hidden_state, lengths=lengths, enforce_sorted=False
        )
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [
                projector(decoder_output) for projector in self.distribution_projector
            ]
        return output, hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            output, _ = self.decode_all(
                input_vector, hidden_state, lengths=decoder_lengths
            )
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            target_scale = apply_to_list(
                target_scale, lambda x: x.repeat_interleave(n_samples, 0)
            )

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, hidden_state = self.decode_all(x, hidden_state)
                prediction = apply_to_list(
                    prediction, lambda x: x[:, 0]
                )  # select first time step
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps
            # to batch_size x time steps x n_samples
            output = apply_to_list(
                output,
                lambda x: x.reshape(-1, n_samples, input_vector.size(1)).permute(
                    0, 2, 1
                ),
            )
        return output

    def forward(
        self, x: dict[str, torch.Tensor], n_samples: int = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward network
        """
        hidden_state = self.encode(x)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(
                    x["encoder_cont"].size(0), device=x["encoder_cont"].device
                ),
                x["encoder_lengths"] - 1,
                self.target_positions.unsqueeze(-1),
            ].T.contiguous(),
        )

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
            n_samples=n_samples,
        )
        # return relevant part
        return self.to_network_output(prediction=output)

    def create_log(self, x, y, out, batch_idx):
        n_samples = [
            self.hparams.n_validation_samples,
            self.hparams.n_plotting_samples,
        ][self.training]
        log = super().create_log(
            x,
            y,
            out,
            batch_idx,
            prediction_kwargs=dict(n_samples=n_samples),
            quantiles_kwargs=dict(n_samples=n_samples),
        )
        return log

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        return_x: bool = False,
        return_y: bool = False,
        mode_kwargs: dict[str, Any] = None,
        trainer_kwargs: Optional[dict[str, Any]] = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        output_dir: Optional[str] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> Prediction:
        """
        predict dataloader

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", "samples" or "raw", or tuple ``("raw", output_name)`` where
                output_name is a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
                dataframe corresponds to the first dimension of the output and the given time index is the time index
                of the first prediction)
            return_decoder_lengths: if to return decoder_lengths (in the same order as the output
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs (in the same order as prediction output)
            return_y: if to return network targets (in the same order as prediction output)
            mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
                for modes "prediction" and "quantiles"
            trainer_kwargs (Dict[str, Any], optional): keyword arguments for the trainer
            write_interval: interval to write predictions to disk
            output_dir: directory to write predictions to. Defaults to None. If set function will return empty list
            n_samples: number of samples to draw. Defaults to 100.

        Returns:
            Prediction: if one of the ```return`` arguments is present,
                prediction tuple with fields ``prediction``, ``x``, ``y``, ``index`` and ``decoder_lengths``
        """  # noqa: E501
        if isinstance(mode, str):
            if mode in ["prediction", "quantiles"]:
                if mode_kwargs is None:
                    mode_kwargs = dict(use_metric=False)
                else:
                    mode_kwargs = deepcopy(mode_kwargs)
                    mode_kwargs["use_metric"] = False
            elif mode == "samples":
                mode = ("raw", "prediction")
        return super().predict(
            data=data,
            mode=mode,
            return_decoder_lengths=return_decoder_lengths,
            return_index=return_index,
            n_samples=n_samples,  # new keyword that is passed to forward function
            return_x=return_x,
            fast_dev_run=fast_dev_run,
            num_workers=num_workers,
            batch_size=batch_size,
            mode_kwargs=mode_kwargs,
            trainer_kwargs=trainer_kwargs,
            write_interval=write_interval,
            output_dir=output_dir,
            return_y=return_y,
            **kwargs,
        )
