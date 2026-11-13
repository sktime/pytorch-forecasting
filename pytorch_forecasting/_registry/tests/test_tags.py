"""Tests for the tag register."""

__author__ = ["echo-xiao"]

from pytorch_forecasting._registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
    OBJECT_TAG_TABLE,
    _BaseTag,
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
