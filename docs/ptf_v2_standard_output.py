import torch
import torch.nn as nn

# 1. Standard Output Contract
# All v2 models must return:
# List[Tensor(batch, time, distribution_params)]
# Each element corresponds to one target / one loss head

class BaseForecastModel(nn.Module):
    def forward(self, x):
        """
        Args:
            x: Tensor(batch, time, features)
        Returns:
            List[Tensor(batch, time, distribution_params)]
        """
        raise NotImplementedError("All models must return a List[Tensor]")

# 2. Example Models


class SimpleForecastModel(BaseForecastModel):
    def __init__(self, input_dim, hidden_dim, distribution_params=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, distribution_params)

    def forward(self, x):
        # x: (batch, time, input_dim)
        h = torch.relu(self.fc(x))
        y = self.out(h)  # (batch, time, distribution_params)
        return [y]       # standardized as List[Tensor]


class MultiTargetForecastModel(BaseForecastModel):
    """
    Example multi-target model with different distribution params per target
    """
    def __init__(self, input_dim, hidden_dim, target_params):
        """
        target_params: list[int], number of distribution params per target
        """
        super().__init__()
        self.shared_fc = nn.Linear(input_dim, hidden_dim)
        self.out_layers = nn.ModuleList([nn.Linear(hidden_dim, p) for p in target_params])

    def forward(self, x):
        h = torch.relu(self.shared_fc(x))
        outputs = [layer(h) for layer in self.out_layers]
        return outputs  # List[Tensor(batch, time, dist_params)]



# 3. Loss Functions


class BaseLoss(nn.Module):
    def forward(self, pred, target):
        raise NotImplementedError


class MAELoss(BaseLoss):
    def forward(self, pred, target):
        # pred, target: Tensor(batch, time, 1)
        return torch.mean(torch.abs(pred - target))


class MSELoss(BaseLoss):
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)


class QuantileLoss(BaseLoss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        # pred: Tensor(batch, time, num_quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - pred[..., i:i+1]
            losses.append(torch.max((q - 1) * err, q * err))
        return torch.mean(torch.stack(losses))


# 4. MultiLoss Wrapper


class MultiLoss(BaseLoss):
    """
    Accepts List[Tensor] predictions and List[Tensor] targets.
    Applies different losses for each target
    """
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, preds, targets):
        if len(preds) != len(targets):
            raise ValueError("Preds and targets list must be the same length")
        total = 0
        for loss_fn, p, t in zip(self.losses, preds, targets):
            total += loss_fn(p, t)
        return total


# 5. Metric Definitions


class BaseMetric:
    def update(self, preds, targets):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class MAEMetric(BaseMetric):
    def __init__(self):
        self.total_error = 0.0
        self.count = 0

    def update(self, preds, targets):
        for p, t in zip(preds, targets):
            self.total_error += torch.sum(torch.abs(p - t)).item()
            self.count += p.numel()

    def compute(self):
        return self.total_error / self.count if self.count > 0 else 0.0


class RMSEMetric(BaseMetric):
    def __init__(self):
        self.total_error = 0.0
        self.count = 0

    def update(self, preds, targets):
        for p, t in zip(preds, targets):
            self.total_error += torch.sum((p - t) ** 2).item()
            self.count += p.numel()

    def compute(self):
        return (self.total_error / self.count) ** 0.5 if self.count > 0 else 0.0



# 6. Utility Functions

def validate_shapes(preds, targets):
    if not isinstance(preds, list) or not isinstance(targets, list):
        raise TypeError("Preds and targets must be List[Tensor]")
    if len(preds) != len(targets):
        raise ValueError("Number of predictions and targets must match")
    for i, (p, t) in enumerate(zip(preds, targets)):
        if p.shape != t.shape:
            raise ValueError(f"Shape mismatch at index {i}: {p.shape} vs {t.shape}")


# 7. Example Vignettes / Usage


if __name__ == "__main__":
    # ----- Single target -----
    x = torch.randn(32, 12, 8)       # batch, time, features
    y_true = [torch.randn(32, 12, 1)]
    model = SimpleForecastModel(input_dim=8, hidden_dim=16)
    y_pred = model(x)

    # Loss & Metric
    loss_fn = MultiLoss([MAELoss()])
    metric = MAEMetric()

    validate_shapes(y_pred, y_true)
    loss = loss_fn(y_pred, y_true)
    metric.update(y_pred, y_true)
    print("Single-target example")
    print("Loss:", loss.item())
    print("MAE:", metric.compute())
    print("-" * 50)

    # Multi-target 
    y_true_multi = [
        torch.randn(32, 12, 1),  # target1
        torch.randn(32, 12, 3)   # target2, e.g., quantile prediction
    ]
    model_multi = MultiTargetForecastModel(input_dim=8, hidden_dim=16, target_params=[1, 3])
    y_pred_multi = model_multi(x)

    loss_fn_multi = MultiLoss([MAELoss(), QuantileLoss(quantiles=[0.1,0.5,0.9])])
    metric_multi = MAEMetric()

    validate_shapes(y_pred_multi, y_true_multi)
    loss_multi = loss_fn_multi(y_pred_multi, y_true_multi)
    metric_multi.update(y_pred_multi, y_true_multi)
    print("Multi-target example")
    print("Loss:", loss_multi.item())
    print("MAE:", metric_multi.compute())
    print("-" * 50)

    # RMSE Metric Example
    metric_rmse = RMSEMetric()
    metric_rmse.update(y_pred_multi, y_true_multi)
    print("RMSE Multi-target example:", metric_rmse.compute())
    print("-" * 50)

    # Probabilistic single-target 
    y_true_quantile = [torch.randn(32, 12, 3)]
    model_quantile = SimpleForecastModel(input_dim=8, hidden_dim=16, distribution_params=3)
    y_pred_quantile = model_quantile(x)

    loss_fn_quantile = MultiLoss([QuantileLoss([0.1, 0.5, 0.9])])
    metric_quantile = MAEMetric()
    validate_shapes(y_pred_quantile, y_true_quantile)
    loss_q = loss_fn_quantile(y_pred_quantile, y_true_quantile)
    metric_quantile.update(y_pred_quantile, y_true_quantile)
    print("Probabilistic example")
    print("Loss:", loss_q.item())
    print("MAE:", metric_quantile.compute())
