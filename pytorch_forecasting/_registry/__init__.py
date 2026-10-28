"""PyTorch Forecasting registry."""

from pytorch_forecasting._registry._lookup import all_objects, all_tags
from pytorch_forecasting._registry._tags import check_tag_is_valid

__all__ = [
    "all_objects",
    "all_tags",
    "check_tag_is_valid",
]
