"""Base Classes for pytorch-forecasting models, skbase compatible for indexing."""

from pytorch_forecasting.utils._dependencies import _safe_import

_SkbaseBaseObject = _safe_import("skbase._base_object._BaseObject")


class _BaseObject(_SkbaseBaseObject):

    pass
