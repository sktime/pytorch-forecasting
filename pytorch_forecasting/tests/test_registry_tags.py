"""Tests for the tag register and tag lookup functionality."""

import ast
from pathlib import Path

import pytest

from pytorch_forecasting._registry import all_tags
from pytorch_forecasting._registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
    check_tag_is_valid,
)


def test_tag_register_type():
    """Test the specification of the tag register. See _tags for specs."""
    assert isinstance(OBJECT_TAG_REGISTER, list)
    assert all(isinstance(tag, tuple) for tag in OBJECT_TAG_REGISTER)

    for tag in OBJECT_TAG_REGISTER:
        assert len(tag) == 4
        assert isinstance(tag[0], str)
        assert isinstance(tag[1], (str, list))
        if isinstance(tag[1], list):
            assert all(isinstance(x, str) for x in tag[1])
        assert isinstance(tag[2], (str, tuple))
        if isinstance(tag[2], tuple):
            assert len(tag[2]) == 2
            assert isinstance(tag[2][0], str)
            assert isinstance(tag[2][1], (list, str))
            if isinstance(tag[2][1], list):
                assert all(isinstance(x, str) for x in tag[2][1])
        assert isinstance(tag[3], str)


def test_check_tag_is_valid():
    """Test that check_tag_is_valid accepts valid and rejects invalid values."""
    check_tag_is_valid("capability:multivariate", True)
    check_tag_is_valid("info:name", "DeepAR")
    check_tag_is_valid("info:compute", 3)
    check_tag_is_valid("authors", ["some-author"])
    check_tag_is_valid("object_type", "metric")
    check_tag_is_valid("info:pred_type", ["point"])

    with pytest.raises(ValueError):
        check_tag_is_valid("capability:multivariate", "not_a_bool")

    with pytest.raises(ValueError):
        check_tag_is_valid("info:compute", "not_an_int")

    with pytest.raises(ValueError):
        check_tag_is_valid("authors", "not_a_list")

    with pytest.raises(ValueError):
        check_tag_is_valid("object_type", "not_a_valid_object_type")

    with pytest.raises(ValueError):
        check_tag_is_valid("info:pred_type", ["not_a_valid_pred_type"])

    with pytest.raises(KeyError):
        check_tag_is_valid("not_a_real_tag", 42)


def test_all_tags_filters_by_object_type():
    """Test that all_tags filters correctly by object_type."""
    metric_tags = all_tags(object_types="metric")
    metric_tag_names = {tag[0] for tag in metric_tags}
    assert "metric_type" in metric_tag_names
    assert "capability:multivariate" not in metric_tag_names

    df = all_tags(as_dataframe=True)
    assert list(df.columns) == ["name", "object_type", "type", "description"]


def _find_tag_keys_in_use():
    """Collect every tag key set in a ``_tags = {...}`` literal in the package."""
    package_root = Path(__file__).parent.parent
    # the tag definition module itself uses "_tags" for its own tag-class schema
    # (tag_name, parent_type, tag_type, short_descr), not for tagging real objects
    registry_tags_file = package_root / "_registry" / "_tags.py"
    tag_keys = set()

    for path in package_root.rglob("*.py"):
        if path == registry_tags_file:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            is_tags_assign = (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_tags"
                and isinstance(node.value, ast.Dict)
            )
            if not is_tags_assign:
                continue
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    tag_keys.add(key.value)

    return tag_keys


def test_all_used_tags_are_registered():
    """Test that every tag used in the package is documented in OBJECT_TAG_LIST.

    Catches tags being added to a ``_pkg.py`` class without a corresponding
    entry in ``pytorch_forecasting/_registry/_tags.py``.
    """
    used_tags = _find_tag_keys_in_use()
    undocumented = used_tags - set(OBJECT_TAG_LIST)

    assert not undocumented, (
        f"The following tags are used in the codebase but not registered in "
        f"pytorch_forecasting/_registry/_tags.py: {sorted(undocumented)}"
    )
