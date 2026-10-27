# Extension templates for PyTorch Forecasting v2

These are copy-and-adapt templates, not importable implementations. Start with
`model.py` and `model_forecaster.py`; use the data-module and dataset templates only
when no existing data module provides the inputs your model needs. The metrics
template is a separate example for adding a metric.

## Model and forecaster

`MyModel` in `model.py` contains the network and implements `forward`. Its `_pkg()`
method returns the associated forecaster class, `MyModelForecaster`; it no longer
returns a separate model-package class.

`MyModelForecaster` in `model_forecaster.py` inherits from `BaseForecaster`. It:

- returns the model class from `get_cls()`;
- returns a compatible concrete data-module class from `get_datamodule_cls()`;
- maps forecaster parameters to model constructor arguments in `get_model_params()`;
- accepts optional trainer, configured data module, and checkpoint arguments using
  the `BaseForecaster` constructor contract.

The forecaster owns the high-level `fit` and `predict` workflow. Both operate on the
public `TimeSeries` data object. Use `TimeSeries` from `pytorch_forecasting.data` to
wrap input data; this template set does not include a separate raw-data/TimeSeries
template.

## Custom data module

Prefer an existing compatible class from `pytorch_forecasting.data.data_module`.
`EncoderDecoderTimeSeriesDataModule` is used as the concrete example in
`model_forecaster.py`; select a class whose batch keys and target format match your
model's `forward`.

When a custom module is needed, subclass `BaseTimeSeriesDataModule`, not
`LightningDataModule` directly. The base class owns data attachment with `with_data`,
metadata access, split setup, and dataloader creation. Implement its data-format hooks:
`_prepare_metadata`, `_preprocess_data`, `_create_windows`, `_build_dataset`, and the
static `collate_fn`. Override the constructor only to add configuration, and pass
`time_series` and supported base options to `super().__init__`.

The private dataset should implement `__len__` and `__getitem__`. Each item is an
`(x, y)` pair; collation turns these into the batch consumed by the model and its loss.
Follow the signatures and tensor format expected by the selected base module and model.

## Metrics

`extension_templates/metrics.py` demonstrates a `MultiHorizonMetric`. Its loss receives
the prediction tensor extracted from the model output and the target tensor; the base
metric handles masking and reduction. Keep the existing metric tensor contract.
Distribution metrics should use the relevant `DistributionLoss` base class and
implement its distribution-specific hooks as needed.

## Validate an extension

Implement test parameters for the forecaster and keep default configurations small.
After placing the model in an importable package and registering its forecaster, run
the v2 estimator suite and the generic estimator checks:

```python
from pytorch_forecasting.utils import check_estimator
from my_package.model import MyModel

check_estimator(MyModel)
```

```powershell
python -m pytest pytorch_forecasting/tests/test_all_v2/test_all_estimators_v2.py
```

Use the v2 integration tests as a guide for fitting, prediction modes, and checkpoint
round trips. These templates do not change the tensor-shape contract.
