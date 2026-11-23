"""Tests for pytorch_forecasting.models.base._base_object."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster_Common


class _DummyModel:
    def __init__(self, x=1):
        self.x = x


class _DummyPkg(_BasePtForecaster_Common):
    @classmethod
    def get_cls(cls):
        return _DummyModel

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"x": 7}


def test_base_pt_forecaster_common_helpers_use_get_cls():
    """Helper methods should instantiate via ``get_cls``."""
    obj = _DummyPkg.create_test_instance()
    objs, names = _DummyPkg.create_test_instances_and_names()

    assert isinstance(obj, _DummyModel)
    assert obj.x == 7
    assert len(objs) == 1
    assert isinstance(objs[0], _DummyModel)
    assert objs[0].x == 7
    assert names == ["_DummyPkg"]
    assert _DummyPkg.name() == "_DummyModel"
