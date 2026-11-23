"""
Chronos foundation model adapter for pytorch-forecasting v2.

Wraps Amazon Chronos (HuggingFace) to expose the BaseForecaster interface,
enabling zero-shot time-series forecasting with the same API as TFT or N-BEATS.

Part of GSoC 2026 - Foundation Model Adapters for pytorch-forecasting v2.
See: https://github.com/sktime/pytorch-forecasting/issues/2051
"""


class ChronosForecaster:
    """
    Adapter wrapping Amazon Chronos for pytorch-forecasting v2.

    Provides zero-shot inference via HuggingFace ChronosPipeline,
    accepting standard TimeSeriesDataSet input and returning
    standard Prediction objects with point and quantile forecasts.

    Parameters
    ----------
    model_name : str
        HuggingFace model name, e.g. 'amazon/chronos-t5-small'
    prediction_length : int
        Number of steps to forecast
    num_samples : int
        Number of samples for probabilistic forecasting (default: 20)
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        prediction_length: int = 12,
        num_samples: int = 20,
    ):
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, data):
        """Run zero-shot inference. Implementation in progress."""
        raise NotImplementedError("Chronos adapter implementation in progress.")