# Custom Network Extension Template (v1)

This folder provides a **minimal template** for adding a new neural network to
`pytorch_forecasting` using the **v1 API**.

It is intended as a starting point for third-party contributors who want to
implement a new model that integrates with the existing testing and metadata
infrastructure of pytorch-forecasting.

## Files

- **`model.py`**  
  Minimal model implementation that inherits from `BaseModel`.  
  Replace the placeholder logic with your own neural network.

- **`package.py`**  
  Package container that exposes metadata (`_tags`) and links to your model
  via `get_cls()`. This is required for CI discovery and testing.

## How to use

1. Copy this folder to `pytorch_forecasting/models/` and rename it for your model.
2. Implement your model logic in `model.py`.
3. Update `_tags` in `package.py` to reflect your model’s actual capabilities.
4. Ensure `get_cls()` returns your model class.

## Notes

- This template targets **v1 only** (it does not cover the v2 API).
- `get_base_test_params()` is **optional** — most models can simply return `[{}]`
  unless they require custom test setup.
- Advanced hooks (custom dataloaders, specialized test fixtures, etc.) can be
  added later as needed.
