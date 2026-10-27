# Extension templates for PyTorch Forecasting v2

These are copy-and-adapt templates, not importable implementations. Start with
`model.py` and `model_forecaster.py`; use the data-module and dataset templates only
when no existing data module provides the inputs your model needs. The metrics
template is a separate example for adding a metric.

## How to use the templates

Copy the required files to an importable package and give the classes and files
descriptive names. Work through every TODO, update the parameter descriptions and
method docstrings, and implement the mandatory methods. Keep the instructions and
TODOs in these shared templates so the next contributor has the same guidance;
replace them with implementation-specific documentation in your adapted classes.

Keep model and data-module imports inside the corresponding class-linkage methods.
Import optional dependencies inside methods so importing an extension does not
eagerly load dependencies that may be unavailable.

## Model and forecaster

`MyModel` in `model.py` contains the network and implements `forward`. Its legacy
`_pkg()` linkage returns the associated forecaster class, `MyModelForecaster`; it
does not return a separate model-package class. `forward()` accepts the input
dictionary produced by the selected datamodule and returns a dictionary containing
the `"prediction"` tensor used by the base model and loss.

`MyModelForecaster` in `model_forecaster.py` inherits from `BaseForecaster`. It:

- returns the model class from `get_cls()`;
- returns a compatible concrete data-module class from `get_datamodule_cls()`;
- maps forecaster parameters to model constructor arguments in `get_model_params()`;
- leaves data-derived `metadata` to `BaseForecaster`, which obtains it from the
  attached datamodule and passes it to the model constructor;
- accepts optional trainer, configured data module, and checkpoint arguments using
  the `BaseForecaster` constructor contract.

The forecaster owns the high-level `fit` and `predict` workflow. Pass a
`TimeSeries` object to `fit()` and `predict()`; the forecaster attaches it to the
configured datamodule and initializes the model with the resulting metadata.
Prediction mode returns a `TimeSeries`; `mode="raw"` returns the model output
dictionary. Use `TimeSeries` from `pytorch_forecasting.data` to wrap input data;
this template set does not include a separate raw-data/TimeSeries template.

### Forecaster configuration

Implement and document the model options in the forecaster constructor, then map
them to the network constructor in `get_model_params()`. Keep trainer, datamodule,
checkpoint settings, and data-derived metadata out of that mapping. Keep
constructor attributes unchanged when resolving defaults.

Set the forecaster's `_tags` to describe the implemented model. Tags are inherited
when not overridden:

| Tag | Meaning |
| --- | --- |
| `info:name` | Model name matching the neural-network class. |
| `info:compute` | Approximate compute cost, from 1 (light) to 5 (heavy). |
| `info:pred_type` | Supported prediction types: `point`, `quantile`, or `distr`. |
| `info:y_type` | Supported target types: `numeric` or `category`. |
| `authors` | Contributor GitHub handles. |
| `python_dependencies` | Additional optional dependencies, or an empty list. |
| `capability:exogenous` | Whether the model uses exogenous covariates. |
| `capability:multivariate` | Whether multiple targets are supported. |
| `capability:pred_int` | Whether prediction intervals are supported. |
| `capability:flexible_history_length` | Whether variable history lengths work. |
| `capability:cold_start` | Whether predictions work without a long history. |

Only advertise capabilities implemented by the adapted model. In
`get_test_train_params()`, keep the first dictionary empty to exercise defaults
and include a second, small configuration to cover a different case. Keep imports
needed only for testing inside that method.

### Network implementation

Call `super().__init__()` with the supported training configuration and use
`save_hyperparameters()` as illustrated in `model.py`. Accept the metadata supplied
by the forecaster and derive required network dimensions from it. Implement
`forward()` for the selected data module's inputs and return the prediction tensor
under the `"prediction"` key. Document the input keys and tensor shapes.

Update `_pkg()` to return the corresponding forecaster. Add documented helper
methods for architecture-specific logic without overriding other private base
methods. Put reusable custom layers in the appropriate layers module when
contributing them to the library.

## Custom data module

Prefer an existing compatible class from `pytorch_forecasting.data.data_module`.
`EncoderDecoderTimeSeriesDataModule` is used as the concrete example in
`model_forecaster.py`; select a class whose batch keys and target format match your
model's `forward`.

When a custom module is needed, subclass `BaseTimeSeriesDataModule`, not
`LightningDataModule` directly. The base class owns data attachment with `with_data`,
metadata access, split setup, and dataloader creation. Implement its data-format hooks:
`_prepare_metadata`, `_preprocess_data`, `_create_windows`, `_build_dataset`, and the
static `collate_fn`. Inherit the base constructor unless the subclass needs additional
configuration. If you override it, expose configurable base options as explicit
parameters rather than hiding them in `**kwargs`, and pass supported base options to
`super().__init__`. Store any subclass-specific options as same-named attributes
before calling `super()`; the base captures constructor parameters this way so
`with_data()` can reconstruct the subclass with its configuration.

The private dataset should implement `__len__` and `__getitem__`. Each item is an
`(x, y)` pair; collation turns these into the batch consumed by the model and its loss.
Follow the signatures and tensor format expected by the selected base module and model.

Document how `_prepare_metadata()` derives each model dimension and feature group,
the transformations performed by `_preprocess_data()`, and the valid windows
enumerated by `_create_windows()`. `_build_dataset()` connects those windows to the
private dataset. Keep `__getitem__()` and `collate_fn()` consistent about sample
keys, target layout, padding, and length information. The inherited `metadata`
property caches the metadata; the base also provides split setup and dataloaders.

## Metrics

`extension_templates/metrics.py` demonstrates a `MultiHorizonMetric`. Its loss receives
the prediction tensor extracted from the model output and the target tensor; the base
metric handles masking and reduction. Keep the existing metric tensor contract.
Distribution metrics should use the relevant `DistributionLoss` base class and
implement its distribution-specific hooks as needed.

## Validate an extension

Implement test parameters for the forecaster and keep default configurations small.
Place the model and forecaster in an importable package, ensure the model's `_pkg()`
returns that forecaster, and expose the forecaster to the v2 estimator checks. Run the
v2 estimator suite and the generic estimator checks:

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
