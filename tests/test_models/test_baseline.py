from helpers import monkeypatch_env

from pytorch_forecasting import Baseline


@monkeypatch_env("torch._C._mps_is_available", False)
def test_integration(multiple_dataloaders_with_covariates):
    dataloader = multiple_dataloaders_with_covariates["val"]
    Baseline().predict(dataloader, fast_dev_run=True)
    repr(Baseline())
