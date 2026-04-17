# Contributing guide

You can find our general contributing guide on our [website](https://www.sktime.net/en/latest/get_involved/contributing.html).

## Implementing New Models (v2)

For contributors looking to implement new models under the v2 architecture, we provide two canonical templates:

- **Model Template**: [pytorch_forecasting/models/model_template.py](file:///Users/sujanyd/Desktop/pytorch-forecasting/pytorch_forecasting/models/model_template.py)
  - Canonical `BaseModel` subclass with detailed docstrings for `__init__`, `forward`, and `training_step`.
- **Package Template**: [pytorch_forecasting/models/pkg_template.py](file:///Users/sujanyd/Desktop/pytorch-forecasting/pytorch_forecasting/models/pkg_template.py)
  - Canonical `Base_pkg` container for model registration and metadata tags.

These files contain detailed documentation on how to implement your model and explain the significance of hyper-parameters and metadata tags. Key methods to implement/override include:
- `forward`: Primary model logic.
- `training_step`: Optional, for custom training logic (e.g., teacher forcing).
- `get_test_train_params`: In the package class, provides parameters for automated testing.
