# Custom Network Extension Template (v1)

This folder provides a minimal extension template for adding a new neural
network to `pytorch_forecasting` using the **v1 API**.

This is **not a working model** and is **not meant to be imported directly**.
It is a coding scaffold that contributors should copy and adapt when adding
new v1 models to the ecosystem.

---

## Purpose

This template exists to:

- Provide a **consistent starting structure** for new v1 models.
- Make explicit **which methods are required vs. optional**.
- Standardize **metadata (`_tags`) and test fixtures** so CI can discover and
  validate new models.
- Reduce confusion for contributors about how v1 models should be structured.

---

## Folder contents

### `model.py`

A minimal neural network template that should:

- Inherit from an appropriate v1 base class (e.g., `BaseModel` or a relevant subclass).
- Put any reusable layers or submodules of the model into a `layers/` folder (e.g., `my_model/layers/`) to keep the code modular and clean.
- Define at least the following required methods:
  - `__init__`
  - `_pkg`
  - `from_dataset`
  - `forward`
- Optional methods (e.g., `to_prediction` or `to_quantiles` for probabilistic/quantile networks) should be removed if not used to keep the final codebase clean. Only implement/uncomment them if custom post-processing, rescaling, or CDF calculations are needed.

> This file should primarily contain **structured comments and pointers**,
> not real working model code.
> Contributors are expected to replace placeholders with their own implementation.

---

### `_model_pkg.py`

A **private package container** that exposes metadata and links to the model class. It **must**:

- Be named with a leading underscore to mark it as private (e.g., `_my_model_pkg.py` for class `MyModel_pkg`), and be placed in the same directory as the model file.
- Define a `_tags` dictionary that correctly describes the model's capabilities.
- Implement:
  - `get_cls()` → returns the actual model class.
  - `get_base_test_params()` → **REQUIRED** test fixtures.
  - `_get_test_dataloaders_from()` → returns train/validation dataloaders for CI tests.

#### About `_tags`

Each tag in the template includes detailed comments explaining:

- What the tag means.
- What valid/possible values are.
- How a contributor should choose them.

At minimum, `_tags` should include:

- `info:name` (human-readable model name matching the class)
- `info:pred_type` (prediction types: e.g. `["point"]`, `["quantile"]`, `["distr"]`)
- `info:y_type` (target type: e.g. `["numeric"]`, `["category"]`)
- `info:compute` (integer representing compute intensity, 1 to 5)
- `authors` (GitHub username list)
- `python_dependencies` (list of external packages if needed)
- `capability:exogenous` (bool: whether model supports exogenous variables)
- `capability:multivariate` (bool: whether model supports multivariate targets)
- `capability:pred_int` (bool: whether model supports prediction intervals)
- `capability:flexible_history_length` (bool: whether model works with variable-length history)
- `capability:cold_start` (bool: whether model makes predictions with little/no history)

The class name of the package container **must match the model name**, e.g.:

- If your model is `ExampleNetwork`, the package class should be `ExampleNetwork_pkg` and the package file must be named `_model_pkg.py`.

---

## How to use this template

1. Copy this folder and rename it for your model (e.g., `my_custom_network/`).
2. Rename the private package file `_model_pkg.py` to match your model name (e.g., `_my_custom_network_pkg.py`).
3. Replace placeholders in `model.py` with your actual implementation (and place any reusable submodules/layers in a `layers/` subdirectory).
4. Update all `_tags` in `_my_custom_network_pkg.py` with accurate metadata.
5. Implement `get_base_test_params()` with **realistic test fixtures**.
6. Implement `_get_test_dataloaders_from()` using the dataset provided by CI.
7. Move your completed model folder into `pytorch_forecasting/models/`.
8. Register your model in `pytorch_forecasting/models/__init__.py`.

---

## Testing requirements (CRITICAL)

### `get_base_test_params()` — REQUIRED

This method **must** return at least **two** different parameter settings that:

- Create a valid model instance.
- Exercise different configurations of the model.
- Run quickly in CI.
- **Test defaults**: The first element in the returned list MUST be an empty dictionary `{}` to verify default model initialization works correctly.

Example (illustrative only):

```python
return [
    {},
    {"hidden_size": 8, "use_exogenous": True},
]
```

---

### `_get_test_dataloaders_from()` — REQUIRED

This method must return valid train/validation dataloaders, typically via
the test data scenarios in `pytorch_forecasting.tests._data_scenarios`:

```python
from pytorch_forecasting.tests._data_scenarios import (
    data_with_covariates,
    make_dataloaders,
)

dwc = data_with_covariates()
dataloaders = make_dataloaders(dwc, target="target", ...)
return dataloaders
```

These dataloaders are used by the CI test suite.

---

## Reference models (look at these)

When in doubt, study existing v1 models such as:

- `DecoderMLP` — simple MLP-based model
- `DeepAR` — autoregressive RNN model
- `TemporalFusionTransformer` — attention-based model
- `NHiTS` — hierarchical interpolation model

They provide good examples of:

- Proper constructor design (`__init__`)
- `from_dataset` factory method
- Forward logic
- Packaging via `_pkg()`
- Test parameter structure

---

## Scope

- This template targets **v1 API only**.
- It does **not** cover v2 models.
- It is meant for contributors adding **new neural networks**, not for
  users training models.
