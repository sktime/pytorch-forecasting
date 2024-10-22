__all__ = ["AutoRegressiveBaseModel"]

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor

from pytorch_forecasting.metrics import DistributionLoss, MultiLoss
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModel as AutoRegressiveBaseModel_
from pytorch_forecasting.utils import apply_to_list, to_list


class AutoRegressiveBaseModel(AutoRegressiveBaseModel_):  # pylint: disable=abstract-method
    """Basically AutoRegressiveBaseModel from `pytorch_forecasting` but fixed for multi-target. Worked for `LSTM`."""

    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Convert network output to rescaled and normalized prediction.
        Function is typically not called directly but via :py:meth:`~decode_autoregressive`.
        Args:
            normalized_prediction_parameters (torch.Tensor): network prediction output
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale to rescale network output
            n_samples (int, optional): Number of samples to draw independently. Defaults to 1.
            **kwargs: extra arguments for dictionary passed to :py:meth:`~transform_output` method.
        Returns:
            Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]: tuple of rescaled prediction and
                normalized prediction (e.g. for input into next auto-regressive step)
        """
        B = normalized_prediction_parameters.size(0)
        D = normalized_prediction_parameters.size(-1)
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        prediction_parameters = self.transform_output(
            prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs
        )

        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            if n_samples > 1:
                prediction_parameters = apply_to_list(
                    prediction_parameters, lambda x: x.reshape(int(x.size(0) / n_samples), n_samples, -1)
                )
                prediction = self.loss.sample(prediction_parameters, 1)
                prediction = apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)
        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
        if isinstance(normalized_prediction, list):
            input_target = normalized_prediction[-1]  # torch.cat(normalized_prediction, dim=-1)  # dim=-1
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction
        assert input_target.size(0) == B
        assert input_target.size(-1) == D, f"{input_target.size()} but D={D}"
        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)
        return prediction, input_target

    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Make predictions in auto-regressive manner. Supports only continuous targets.
        Args:
            decode_one (Callable):
                function that takes at least the following arguments:
                * ``idx`` (int): index of decoding step (from 0 to n_decoder_steps-1)
                * ``lagged_targets`` (List[torch.Tensor]): list of normalized targets.
                    List is ``idx + 1`` elements long with the most recent entry at the end, i.e.
                    ``previous_target = lagged_targets[-1]`` and in general ``lagged_targets[-lag]``.
                * ``hidden_state`` (Any): Current hidden state required for prediction. Keys are variable
                names. Only lags that are greater than ``idx`` are included.
                * additional arguments are not dynamic but can be passed via the ``**kwargs`` argument And
                returns tuple of (not rescaled) network prediction output and hidden state for next
                auto-regressive step.
            first_target (Union[List[torch.Tensor], torch.Tensor]): first target value to use for decoding
            first_hidden_state (Any): first hidden state used for decoding
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale as in ``x``
            n_decoder_steps (int): number of decoding/prediction steps
            n_samples (int): number of independent samples to draw from the distribution -
                only relevant for multivariate models. Defaults to 1.
            **kwargs: additional arguments that are passed to the decode_one function.
        Returns:
            Union[List[torch.Tensor], torch.Tensor]: re-scaled prediction
        """
        # make predictions which are fed into next step
        output: List[Union[List[Tensor], Tensor]] = []
        current_hidden_state = first_hidden_state
        normalized_output = [first_target]
        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )
            assert isinstance(current_target, Tensor)
            # get prediction and its normalized version for the next step
            prediction, current_target = self.output_to_prediction(
                current_target, target_scale=target_scale, n_samples=n_samples
            )

            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples
            output.append(prediction)

        if isinstance(self.hparams.target, str):
            # Here, output is List[Tensor]
            final_output = torch.stack(output, dim=1)  # type: ignore
            return final_output
        # For multi-targets: output is List[List[Tensor]]
        # final_output_multitarget = [
        #     torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))
        # ]
        # self.target_positions is always Tensor([0]), so len() of that is always 1...
        final_output_multitarget = torch.stack([out[0] for out in output], dim=1)
        if final_output_multitarget.dim() > 3:
            final_output_multitarget = final_output_multitarget.squeeze(2)

        r = [final_output_multitarget[..., i] for i in range(final_output_multitarget.size(-1))]
        return r
