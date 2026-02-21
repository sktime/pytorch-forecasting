"""Tests for the _coerce utility functions."""
import pytest

from pytorch_forecasting.utils._coerce import _coerce_to_dict, _coerce_to_list


def test_coerce_to_list_none():
    assert _coerce_to_list(None) == []

def test_coerce_to_list_str():
    assert _coerce_to_list("abc") == ["abc"]

def test_coerce_to_list_list():
    assert _coerce_to_list([1, 2, 3]) == [1, 2, 3]

def test_coerce_to_list_tuple():
    assert _coerce_to_list((1, 2)) == [1, 2]

def test_coerce_to_list_set():
    assert sorted(_coerce_to_list({3, 1, 2})) == [1, 2, 3]

def test_coerce_to_list_generator():
    gen = (x for x in range(3))
    assert _coerce_to_list(gen) == [0, 1, 2]

def test_coerce_to_dict_none():
    assert _coerce_to_dict(None) == {}

def test_coerce_to_dict_dict():
    d = {"a": 1}
    result = _coerce_to_dict(d)
    assert result == d
    assert result is not d  # should be a deepcopy

def test_coerce_to_dict_nested():
    d = {"a": {"b": 2}}
    result = _coerce_to_dict(d)
    assert result == d
    assert result is not d
    assert result["a"] == d["a"]
    assert result["a"] is not d["a"]  # nested dict should also be a deepcopy

def test_coerce_to_dict_other():
    class Dummy:
        pass
    obj = Dummy()
    result = _coerce_to_dict(obj)
    assert result == obj or isinstance(result, Dummy)

def test_coerce_to_list_type_assertions():
    assert isinstance(_coerce_to_list(None), list)
    assert isinstance(_coerce_to_list("abc"), list)
    assert isinstance(_coerce_to_list([1, 2]), list)
    assert isinstance(_coerce_to_list((1, 2)), list)
    assert isinstance(_coerce_to_list({1, 2}), list)
    assert isinstance(_coerce_to_list(x for x in range(2)), list)

def test_coerce_to_dict_type_assertions():
    assert isinstance(_coerce_to_dict(None), dict)
    assert isinstance(_coerce_to_dict({}), dict)
