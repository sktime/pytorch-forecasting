from pytorch_forecasting.data.encoders import GroupNormalizer
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
    MultivariateNormalDistributionLoss,
    NegativeBinomialDistributionLoss,
    NormalDistributionLoss,
    PoissonLoss,
    QuantileLoss,
    TweedieLoss,
)

POINT_LOSSES = [
    MAE(),
    RMSE(),
    SMAPE(),
    MAPE(),
    PoissonLoss(),
    CrossEntropy(),
    MASE(),
    TweedieLoss(),
]
QUANTILE_LOSSES = [QuantileLoss()]
DISTR_LOSSES = [
    NormalDistributionLoss(),
    NegativeBinomialDistributionLoss(),
    MultivariateNormalDistributionLoss(),
    LogNormalDistributionLoss(),
    BetaDistributionLoss(),
    ImplicitQuantileNetworkDistributionLoss(),
]

ALL_LOSSES_BY_TYPE = {
    "point": POINT_LOSSES,
    "quantile": QUANTILE_LOSSES,
    "distr": DISTR_LOSSES,
}


LOSS_SPECIFIC_PARAMS = {
    "BetaDistributionLoss": {
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="logit"
            )
        },
    },
    "LogNormalDistributionLoss": {
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        },
    },
    "NegativeBinomialDistributionLoss": {
        "clip_target": False,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(groups=["agency", "sku"], center=False)
        },
    },
    "MultivariateNormalDistributionLoss": {
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        },
    },
}
