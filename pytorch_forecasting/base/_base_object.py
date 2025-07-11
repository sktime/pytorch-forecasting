"""Base object class for pytorch-forecasting metrics."""

from pytorch_forecasting.utils._dependencies import _safe_import

_SkbaseBaseObject = _safe_import("skbase.base.BaseObject", pkg_name="scikit-base")

__all__ = ["_BaseObject"]


class _BaseObject(_SkbaseBaseObject):
    pass
