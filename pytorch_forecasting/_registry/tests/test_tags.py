"""Tests for the tag register."""

__author__ = ["echo-xiao"]

import pytest

from pytorch_forecasting._registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
    OBJECT_TAG_TABLE,
    _BaseTag,
    check_tag_is_valid,
)


def test_register_is_populated():
    """Every register row is a 4-tuple and the table matches it."""
    assert len(OBJECT_TAG_REGISTER) > 0
    for row in OBJECT_TAG_REGISTER:
        assert len(row) == 4
        tag_name, parent_type, _, short_descr = row
        assert isinstance(tag_name, str) and tag_name
        assert isinstance(parent_type, str) and parent_type
        assert isinstance(short_descr, str) and short_descr
    assert OBJECT_TAG_TABLE.shape[0] == len(OBJECT_TAG_REGISTER)
    assert set(OBJECT_TAG_LIST) == {row[0] for row in OBJECT_TAG_REGISTER}


def test_base_tag_itself_not_registered():
    """``_BaseTag`` is scaffolding, not a tag."""
    assert "" not in OBJECT_TAG_LIST
    assert _BaseTag.get_class_tags()["tag_name"] == ""


@pytest.mark.parametrize(
    "tag_name, tag_value",
    [
        ("object_type", "metric"),
        ("object_type", ["forecaster_pytorch", "forecaster_pytorch_v1"]),
        ("metric_type", "point"),
        ("capability:exogenous", True),
    ],
)
def test_check_tag_is_valid_accepts(tag_name, tag_value):
    """Valid values pass without raising."""
    check_tag_is_valid(tag_name, tag_value)


@pytest.mark.parametrize(
    "tag_name, tag_value, error",
    [
        ("not_a_tag", "whatever", KeyError),
        ("object_type", "not_an_object_type", ValueError),
        ("metric_type", "not_a_metric_type", ValueError),
        ("metric_type", 42, ValueError),
        ("capability:exogenous", "yes", ValueError),
    ],
)
def test_check_tag_is_valid_rejects(tag_name, tag_value, error):
    """Invalid names raise ``KeyError`` and invalid values raise ``ValueError``."""
    with pytest.raises(error):
        check_tag_is_valid(tag_name, tag_value)


@pytest.mark.parametrize("row", OBJECT_TAG_REGISTER, ids=lambda r: r[0])
def test_every_tag_type_is_supported(row):
    """``check_tag_is_valid`` has an arm for every shape the register declares.

    This is what lets ``check_tag_is_valid`` end without an unreachable
    fallback branch.
    """
    tag_type = row[2]
    if isinstance(tag_type, str):
        assert tag_type in ("bool", "int", "str")
        return
    assert isinstance(tag_type, tuple) and len(tag_type) == 2
    kind, allowed = tag_type
    assert kind in ("str", "list")
    if kind == "str":
        assert isinstance(allowed, list) and all(isinstance(a, str) for a in allowed)
    else:
        assert allowed == "str" or (
            isinstance(allowed, list) and all(isinstance(a, str) for a in allowed)
        )


@pytest.mark.parametrize("row", OBJECT_TAG_REGISTER, ids=lambda r: r[0])
def test_every_tag_is_documented(row):
    """Each tag class carries the sections asked for in review of #2334."""
    tag_name = row[0]
    cls = next(
        c
        for c in _BaseTag.__subclasses__()
        if c.get_class_tags()["tag_name"] == tag_name
    )
    doc = cls.__doc__
    assert doc is not None, f"{tag_name} has no docstring"
    assert "Possible values" in doc, f"{tag_name} lacks a Possible values section"
    assert "Effect" in doc, f"{tag_name} lacks an Effect section"
    assert cls.get_class_tags()["short_descr"], f"{tag_name} has an empty short_descr"


def test_tag_classes_are_not_returned_by_all_objects():
    """Tag classes inherit ``_BaseObject`` and must not pollute the object lookup."""
    from pytorch_forecasting._registry import all_objects

    found = all_objects(return_names=False)
    assert not any(
        getattr(obj, "get_class_tags", lambda: {})().get("object_type") == "tag"
        for obj in found
    )


def _all_declared_tags():
    """Yield (class, tag_name, tag_value) for every tag declared in the package."""
    from pytorch_forecasting._registry import all_objects

    for obj in all_objects(return_names=False):
        for tag_name, tag_value in obj.get_class_tags().items():
            yield obj, tag_name, tag_value


def test_every_declared_tag_is_registered():
    """A tag used in the package but missing from the register fails here."""
    unregistered = {
        tag_name
        for _, tag_name, _ in _all_declared_tags()
        if tag_name not in OBJECT_TAG_LIST
    }
    assert not unregistered, f"tags used but not registered: {sorted(unregistered)}"


def test_every_declared_tag_value_is_valid():
    """A value outside its declared tag_type fails here."""
    failures = []
    for obj, tag_name, tag_value in _all_declared_tags():
        if tag_name not in OBJECT_TAG_LIST:
            continue
        try:
            check_tag_is_valid(tag_name, tag_value)
        except (KeyError, ValueError) as err:
            failures.append(f"{obj.__name__}.{tag_name}: {err}")
    assert not failures, "invalid tag values:\n" + "\n".join(failures)


def test_all_tags_default_returns_register_rows():
    """The default return is the register itself."""
    from pytorch_forecasting._registry import all_tags

    assert sorted(all_tags()) == sorted(OBJECT_TAG_REGISTER)


def test_all_tags_names_only():
    """``return_names=False`` gives tag names without duplicates."""
    from pytorch_forecasting._registry import all_tags

    assert sorted(all_tags(return_names=False)) == sorted(OBJECT_TAG_LIST)


def test_all_tags_filters_by_parent_type():
    """Expected values are written out on purpose: this is a tripwire.

    Adding a scaler strategy tag should fail here, as a reminder to document
    it in the register rather than only on the strategy class.
    """
    from pytorch_forecasting._registry import all_tags

    result = all_tags(parent_types="scaler_strategy", return_names=False)
    assert sorted(result) == ["fit_per_sequence", "is_label_encoder"]


def test_all_tags_as_dataframe():
    """The dataframe form names its columns."""
    import pandas as pd

    from pytorch_forecasting._registry import all_tags

    result = all_tags(as_dataframe=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["name", "scitype", "type", "description"]
    assert len(result) == len(OBJECT_TAG_REGISTER)


def test_all_objects_rejects_unknown_filter_tag():
    """A misspelt tag name is an error, not an empty result."""
    from pytorch_forecasting._registry import all_objects

    with pytest.raises(KeyError) as excinfo:
        all_objects(filter_tags={"not_a_real_tag": True})
    assert "not_a_real_tag" in str(excinfo.value)


def test_all_objects_suggests_a_close_tag_name():
    """The error names the tag the user probably meant."""
    from pytorch_forecasting._registry import all_objects

    with pytest.raises(KeyError) as excinfo:
        all_objects(filter_tags={"capability:exogenus": True})
    assert "capability:exogenous" in str(excinfo.value)


def test_all_objects_rejects_unknown_filter_tag_given_as_str():
    """``filter_tags`` may be a bare str, which is validated the same way."""
    from pytorch_forecasting._registry import all_objects

    with pytest.raises(KeyError):
        all_objects(filter_tags="not_a_real_tag")


def test_all_objects_accepts_known_filter_tags():
    """Validation does not break the normal path."""
    from pytorch_forecasting._registry import all_objects

    result = all_objects(filter_tags={"object_type": "metric"}, return_names=False)
    assert len(result) > 0
