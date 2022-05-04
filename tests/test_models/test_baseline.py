from pytorch_forecasting import Baseline


def test_integration(multiple_dataloaders_with_covariates):
    dataloader = multiple_dataloaders_with_covariates["val"]
    Baseline().predict(dataloader, fast_dev_run=True)
    repr(Baseline())
