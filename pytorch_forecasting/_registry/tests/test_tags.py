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


def test_tag_register_type():
    """Test the specification of the tag register. See _tags for specs."""
    assert isinstance(OBJECT_TAG_REGISTER, list)
    assert len(OBJECT_TAG_REGISTER) > 0
    assert all(isinstance(tag, tuple) for tag in OBJECT_TAG_REGISTER)

    # _BaseTag is scaffolding, not a tag, and must not appear in the register
    assert "" not in OBJECT_TAG_LIST
    assert _BaseTag.get_class_tags()["tag_name"] == ""

    for tag in OBJECT_TAG_REGISTER:
        assert len(tag) == 4
        tag_name, parent_type, tag_type, short_descr = tag

        assert isinstance(tag_name, str) and tag_name
        assert isinstance(parent_type, str) and parent_type
        assert isinstance(short_descr, str) and short_descr

        # tag_type is either a plain type name, or (kind, allowed values)
        assert isinstance(tag_type, (str, tuple))
        if isinstance(tag_type, str):
            assert tag_type in ("bool", "int", "str")
            continue

        assert len(tag_type) == 2
        kind, allowed = tag_type
        assert kind in ("str", "list")
        if kind == "str":
            assert isinstance(allowed, list)
            assert all(isinstance(a, str) for a in allowed)
        else:
            assert allowed == "str" or (
                isinstance(allowed, list) and all(isinstance(a, str) for a in allowed)
            )

    assert OBJECT_TAG_TABLE.shape[0] == len(OBJECT_TAG_REGISTER)
    assert set(OBJECT_TAG_LIST) == {tag[0] for tag in OBJECT_TAG_REGISTER}


def test_every_tag_is_documented():
    """Each tag class carries the sections asked for in the review of #2334.

    That review asked every tag to document its possible values, its default and
    what each value changes. Asserting it here keeps the standard from degrading
    silently as tags are added.
    """
    undocumented = []

    for tag_name in OBJECT_TAG_LIST:
        cls = next(
            c
            for c in _BaseTag.__subclasses__()
            if c.get_class_tags()["tag_name"] == tag_name
        )
        doc = cls.__doc__
        if doc is None:
            undocumented.append(f"{tag_name}: no docstring")
            continue
        missing = [s for s in ("Possible values", "Effect") if s not in doc]
        if missing:
            undocumented.append(f"{tag_name}: missing {', '.join(missing)}")
        if not cls.get_class_tags()["short_descr"]:
            undocumented.append(f"{tag_name}: empty short_descr")

    assert not undocumented, "under-documented tags:\n" + "\n".join(undocumented)


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


def test_object_type_enum_covers_test_class_registry():
    """Every scitype that dispatches to a test class must be a registered value."""
    from pytorch_forecasting.tests.test_class_register import get_test_class_registry

    registered = OBJECT_TAG_TABLE[OBJECT_TAG_TABLE[0] == "object_type"].iloc[0, 2][1]
    for scitype in get_test_class_registry():
        assert scitype in registered, (
            f"{scitype!r} dispatches to a test class but is not a registered "
            "object_type value"
        )
