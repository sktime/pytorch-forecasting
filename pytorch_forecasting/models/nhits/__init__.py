"""N-HiTS model for timeseries forecasting with covariates."""

from pytorch_forecasting.models.nhits._nhits import NHiTS
from pytorch_forecasting.models.nhits.sub_modules import NHiTS as NHiTSModule

__all__ = ["NHits", "NHiTSModule"]
