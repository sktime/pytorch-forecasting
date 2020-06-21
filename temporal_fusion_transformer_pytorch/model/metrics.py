from typing import List

import torch
from torch import nn
from torch.nn.utils import rnn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float], cummulative: bool = False, log_target: bool = None):
        """
        Quantile loss

        Args:
            quantiles (List[float]): quantiles to predict (between 0 and 1)
            cummulative (bool): if loss should be calculated cummulatively, i.e.
                if false, the quantiles hold true for individual predictions but
                if true, the quantiles hold true if the predictions are cummulatively
                summed. This is useful if total quantities over the prediction horizon
                are supposed to be predicted.
        """
        super().__init__()
        self.quantiles = quantiles
        self.cummulative = cummulative
        self.log_target = log_target
        if self.cummulative:
            assert self.log_target is not None

    def forward(self, preds, target):
        if isinstance(target, rnn.PackedSequence):
            target, lengths = rnn.pad_packed_sequence(target, batch_first=True)
        else:
            lengths = torch.LongTensor([target.size(1)]).expand(target.size(0))

        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        if self.cummulative:
            if self.log_target:
                preds = preds.exp().cumsum(dim=-2).log()
                target = target.exp().cumsum(dim=-1).log()
            else:
                preds = preds.cumsum(dim=-2)
                target = target.cumsum(dim=-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        losses = torch.cat(losses, dim=1)
        mask = torch.arange(target.size(1)).unsqueeze(0) >= lengths.unsqueeze(-1)
        losses = losses.masked_fill(mask.unsqueeze(1), 0.0)
        quantile_losses = torch.sum(losses, dim=1)
        loss = torch.mean(quantile_losses)
        return loss
