import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles, cummulative=False):
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

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        if self.cummulative:
            preds = preds.cumsum(dim=-2)
            target = target.cumsum(dim=-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        quantile_losses = torch.sum(torch.cat(losses, dim=1), dim=1)
        loss = torch.mean(quantile_losses)
        return loss
