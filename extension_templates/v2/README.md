# Extension Templates for PyTorch Forecasting (v2)

This folder contains implementation templates for quick implementation of new estimators and data modules.
**These are NOT base classes or concrete classes to import. They are "fill-in" coding templates.**

**How to use:**
Make a copy of the template in a suitable location, give it a descriptive name, and fill in the mandatory methods designated by "todo" comments.

---

## Architecture Overview

The v2 architecture strictly separates ML logic from framework metadata to allow lazy loading and lower memory footprints.

### 1. Model and Package (`model.py` and `model_pkg.py`)

* **Model Class (`MyModel`)**: Contains the PyTorch neural network, forward passes, and training logic.
* **Package Class (`MyModel_pkg`)**: Contains metadata, tags, capabilities, and test parameters.
* **Connection**:
* `MyModel._pkg()` returns the package class.
* `MyModel_pkg.get_cls()` returns the model class.



### 2. Data Module and Dataset (`data_module.py` and `_dataset.py`)

* **Data Module**: Manages the ML pipeline setup, data splits, and dataset metadata extraction. Inherits from `LightningDataModule`.
* **Dataset**: Private PyTorch `Dataset` that handles `__getitem__` logic for the dataloaders.

---

## Implementing a New Model

Copy `model.py` and `model_pkg.py` to your target directory.

### Package Configuration (`model_pkg.py`)

Inherits from `Base_pkg`. Implement the following:

**Tags (`_tags`):**
Dictionary defining framework integration rules. Tags are inherited from parent class if they are not set.

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

**Mandatory Methods:**

* `get_cls()`: Imports and returns `MyModel`.
* `get_datamodule_cls()`: Imports and returns the compatible PyTorch Forecasting datamodule.
  * **Determining compatibility:** Inspect the available classes in the [`pytorch_forecasting.data.data_module`](https://github.com/sktime/pytorch-forecasting/tree/main/pytorch_forecasting/data/data_module) directory. Select the data module that handles the data structures and outputs the tensor keys your model's forward pass expects. Every model must link to at least one compatible data module class.
  * If there is no data module that serves your purpose for your model, you might need to implement a new data module. Please look at [Implementing a New Data Module](https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/README.md#implementing-a-new-data-module) for more info.
* `get_test_train_params()`: Returns a list of dicts for CI testing. The first element must be an empty dict `{}` to test defaults. Ensure test configurations yield low-compute models to prevent timeouts.

### Model Configuration (`model.py`)

Inherits from `BaseModel`. Implement the following:

**Mandatory Methods:**

* `__init__()`: Initialize network components. Must call `self.save_hyperparameters()` and `super().__init__()`.
* `_pkg()`: Class method that imports and returns `MyModel_pkg`.
* `forward(x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`: PyTorch forward pass.

---

## Implementing a New Data Module

#### When to Implement a Custom Data Module
Do not create a new data module if an existing one in [`pytorch_forecasting.data.data_module`](https://github.com/sktime/pytorch-forecasting/tree/main/pytorch_forecasting/data/data_module) can format your data into the required inputs.
Implement a custom data module only when your model requires:

* Unique data structures or non-standard time-series features that existing modules cannot parse.
* Custom metadata preparation steps (`_prepare_metadata`) to configure model architecture shapes.
* Specialized sample-level preprocessing logic (`_preprocess_data`) or batch-collating mechanisms.

If needed, copy data_module.py and _dataset.py to your target directory.

### Data Module (`data_module.py`)

Inherits from `LightningDataModule`. Implement the following:

**Mandatory Methods:**

* `_prepare_metadata()`: Derives the metadata required for model initialization from the raw data/parameters.
* `metadata`: Property that returns `_metadata`, invoking `_prepare_metadata()` if null.
* `_preprocess_data(series_idx)`: Contains logic to transform raw series before dataset consumption.
* `setup(stage: str)`: Standard Lightning method to instantiate train/val/test splits.

### Dataset (`_dataset.py`)

Inherits from `torch.utils.data.Dataset`. Implement the following:

**Mandatory Methods:**

* `__init__(data_module, ...)`: Accepts the parent Data Module to read preprocessed data and states.
* `__getitem__(idx)`: Returns the processed item dict exactly as required by the model's `forward` pass.

---

## Interface Verification

Ensure standard compatibility by running the builtin checks after implementation:

```python
from pytorch_forecasting.utils._estimator_checks import check_estimator
from pytorch_forecasting.models,my_model import MyModel

check_estimator(MyModel)

```
