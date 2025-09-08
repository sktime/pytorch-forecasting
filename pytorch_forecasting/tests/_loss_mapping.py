from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    BetaDistributionLoss,
    CrossEntropy,
    ImplicitQuantileNetworkDistributionLoss,
    LogNormalDistributionLoss,
    MQF2DistributionLoss,
    MultivariateNormalDistributionLoss,
    NegativeBinomialDistributionLoss,
    NormalDistributionLoss,
    PoissonLoss,
    QuantileLoss,
    TweedieLoss,
)

from pytorch_forecasting.metrics._distributions_pkg import (
    BetaDistributionLoss_pkg,
    ImplicitQuantileNetworkDistributionLoss_pkg,
    LogNormalDistributionLoss_pkg,
    MQF2DistributionLoss_pkg,
    MultivariateNormalDistributionLoss_pkg,
    NegativeBinomialDistributionLoss_pkg,
    NormalDistributionLoss_pkg,
)

# Remove legacy lists and mappings for losses by pred/y type and tensor shape checks.
# Use tags and _get_test_dataloaders_from for all compatibility and test setup.

METRIC_PKGS = [
    BetaDistributionLoss_pkg,
    NegativeBinomialDistributionLoss_pkg,
    MultivariateNormalDistributionLoss_pkg,
    LogNormalDistributionLoss_pkg, 
    NormalDistributionLoss_pkg,
    ImplicitQuantileNetworkDistributionLoss_pkg
]

LOSS_SPECIFIC_PARAMS = {
    pkg._tags.get("info:metric_name", pkg.__name__.replace("_pkg", "")): {
        k: v for k, v in pkg._tags.items() if k not in ["metric_type", "distribution_type", "info:metric_name", "requires:data_type"]
    }
    for pkg in METRIC_PKGS
}

def get_compatible_losses(pred_types, y_types):
    """
    Get compatible losses based on prediction types and target types.
    """
    compatible_losses = []
    for pkg in METRIC_PKGS:
        pkg_pred_types = pkg._tags.get("compatible_pred_types", [])
        pkg_y_types = pkg._tags.get("compatible_y_types", [])
        if any(pt in pred_types for pt in pkg_pred_types) and any(yt in y_types for yt in pkg_y_types):
            compatible_losses.append(pkg.get_cls()())
    return compatible_losses

def get_test_dataloaders_for_loss(pkg, params=None):
    """
    Get test dataloaders for a given loss package using its tags and method.
    """
    return pkg._get_test_dataloaders_from(params or {})

def check_loss_output_shape(pkg, y_pred, y_true):
    """
    Check that the output shape of the loss matches the expected shape from tags.
    """
    expected_ndim = pkg._tags.get("expected_loss_ndim", None)
    loss_instance = pkg.get_cls()()
    result = loss_instance(y_pred, y_true)
    if expected_ndim is not None:
        assert result.ndim == expected_ndim
