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
- Define at least the following methods:

  - `__init__`
  - `_pkg`
  - `from_dataset`
  - `forward`
  - `to_prediction`
  - `to_quantiles` (if probabilistic outputs are supported)

> This file should primarily contain **structured comments and pointers**,
> not real working model code.
> Contributors are expected to replace placeholders with their own implementation.

---

### `package.py`

A **package container** that exposes metadata and links to the model class. It **must**:

- Define a `_tags` dictionary that correctly describes the model’s capabilities.
- Implement:

  - `get_cls()` → returns the actual model class.
  - `get_base_test_params()` → **REQUIRED** test fixtures.
  - `_get_test_dataloaders_from()` → returns train/validation dataloaders for CI tests.

#### About `_tags`

Each tag in the template should include a comment explaining:

- What the tag means.
- What valid values are.
- How a contributor should choose them.

At minimum, `_tags` should include (with comments in the template):

- `info:name`
- `info:pred_type`
- `info:y_type`
- `capability:exogenous`
- `info:compute`
- `authors`

The class name of the package container **must match the model name**, e.g.:

- If your model is `ExampleNetwork`, the package class should be `ExampleNetwork_pkg`.

---

## How to use this template

1. Copy this folder into the **top-level `extension_templates/` directory**
   and rename it for your model.
2. Replace placeholders in `model.py` with your actual implementation.
3. Update all `_tags` in `package.py` with accurate metadata.
4. Implement `get_base_test_params()` with **realistic test fixtures**.
5. Implement `_get_test_dataloaders_from()` using the dataset provided by CI.

---

## Testing requirements (CRITICAL)

### `get_base_test_params()` — REQUIRED

This method **must** return at least **two** different parameter settings that:

- Create a valid model instance.
- Exercise different configurations of the model.
- Run quickly in CI.

Example (illustrative only):

```python
return [
    {"hidden_size": 8},
    {"hidden_size": 16, "use_exogenous": True},
]
```

---

### `_get_test_dataloaders_from()` — REQUIRED

This method must return valid train/validation dataloaders, typically via:
---

## Reference models (look at these)

When in doubt, study existing v1 models such as:

- `DeepAR`
- `TemporalFusionTransformer`
- `NHiTS`

They provide good examples of:

- Proper constructor design (`__init__`)
- `from_dataset`
- Forward logic
- Packaging via `_pkg()`
- Test parameter structure

---

## Scope

- This template targets **v1 API only**.
- It does **not** cover v2 models.
- It is meant for contributors adding **new neural networks**, not for users training models.

```python
return dataset.to_dataloaders(batch_size=16)
```

These dataloaders are used by the CI test suite.
