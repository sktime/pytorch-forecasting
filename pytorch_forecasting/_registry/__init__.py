"""PyTorch Forecasting registry."""

from pytorch_forecasting._registry._lookup import all_objects, all_tags
from pytorch_forecasting._registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
    check_tag_is_valid,
)

__all__ = [
    "OBJECT_TAG_LIST",
    "OBJECT_TAG_REGISTER",
    "all_objects",
    "all_tags",
    "check_tag_is_valid",
]
