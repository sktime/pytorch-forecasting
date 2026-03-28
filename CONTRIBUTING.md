# Contributing guide

You can find our general contributing guide on our [website](https://www.sktime.net/en/latest/get_involved/contributing.html).

## Implementing New Models (v2)

For contributors looking to implement new models under the v2 architecture, we provide a consolidated template:

- **V2 Model Template**: [pytorch_forecasting/models/v2_template.py](file:///Users/sujanyd/Desktop/pytorch-forecasting/pytorch_forecasting/models/v2_template.py)

This file contains both the `BaseModel` and `Base_pkg` subclasses needed to register and test your model. Follow the `CONTRIBUTOR TODO` comments in the file. Key methods include:

Follow the `CONTRIBUTOR TODO` comments in these files to implement your model. Key methods to implement/override include:
- `forward`: Primary model logic.
- `training_step`: Optional, for custom training logic (e.g., teacher forcing).
- `get_test_train_params`: In the package class, provides parameters for automated testing.
