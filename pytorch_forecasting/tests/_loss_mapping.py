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

POINT_LOSSES_NUMERIC = [
    MAE(),
    RMSE(),
    SMAPE(),
    MAPE(),
    PoissonLoss(),
    MASE(),
    TweedieLoss(),
]

POINT_LOSSES_CATEGORY = [
    CrossEntropy(),
]

QUANTILE_LOSSES_NUMERIC = [
    QuantileLoss(),
]

DISTR_LOSSES_NUMERIC = [
    NormalDistributionLoss(),
    NegativeBinomialDistributionLoss(),
    MultivariateNormalDistributionLoss(),
    LogNormalDistributionLoss(),
    BetaDistributionLoss(),
    ImplicitQuantileNetworkDistributionLoss(),
    # todo: still need some debugging to add the MQF2DistributionLoss
]

METRIC_PKGS = [
    BetaDistributionLoss_pkg,
    NegativeBinomialDistributionLoss_pkg,
    MultivariateNormalDistributionLoss_pkg,
    LogNormalDistributionLoss_pkg, 
    NormalDistributionLoss_pkg,
    ImplicitQuantileNetworkDistributionLoss_pkg
]

LOSSES_BY_PRED_AND_Y_TYPE = {
    ("point", "numeric"): POINT_LOSSES_NUMERIC,
    ("point", "category"): POINT_LOSSES_CATEGORY,
    ("quantile", "numeric"): QUANTILE_LOSSES_NUMERIC,
    ("quantile", "category"): [],
    ("distr", "numeric"): DISTR_LOSSES_NUMERIC,
    ("distr", "category"): [],
}

LOSS_SPECIFIC_PARAMS = {
    pkg._tags.get("info:metric_name", pkg.__name__.replace("_pkg", "")): {
        k: v for k, v in pkg._tags.items() if k not in ["metric_type", "distribution_type", "info:metric_name", "requires:data_type"]
    }
    for pkg in METRIC_PKGS
}


def get_compatible_losses(pred_types, y_types):
    """Get compatible losses based on prediction types and target types.

    Parameters
    ----------
    pred_types : list of str
        Prediction types, e.g., ["point", "distr"]
    y_types : list of str
        Target types, e.g., ["numeric", "category"]

    Returns
    -------
    list
        List of compatible loss instances
    """
    compatible_losses = []

    for pred_type in pred_types:
        for y_type in y_types:
            key = (pred_type, y_type)
            if key in LOSSES_BY_PRED_AND_Y_TYPE:
                compatible_losses.extend(LOSSES_BY_PRED_AND_Y_TYPE[key])

    return compatible_losses