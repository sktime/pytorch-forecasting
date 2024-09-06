from helpers import monkey_patch_torch_fn

from pytorch_forecasting import Baseline


@monkey_patch_torch_fn("torch._C._mps_is_available", False)
def test_integration(multiple_dataloaders_with_covariates):
    dataloader = multiple_dataloaders_with_covariates["val"]
    Baseline().predict(dataloader, fast_dev_run=True)
    repr(Baseline())
