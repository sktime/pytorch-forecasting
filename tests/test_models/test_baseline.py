from pytorch_forecasting import Baseline


def test_integration(multiple_dataloaders_with_covariates):
    dataloader = multiple_dataloaders_with_covariates["val"]
    Baseline().predict(dataloader)
