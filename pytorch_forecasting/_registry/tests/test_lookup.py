"""Tests for the registry lookup utilities."""

__author__ = ["echo-xiao"]

import pytest

from pytorch_forecasting._registry import all_objects, all_tags
from pytorch_forecasting._registry._tags import OBJECT_TAG_LIST, OBJECT_TAG_REGISTER


def test_all_tags():
    """Check that all_tags return arguments have the correct type and shape."""
    import pandas as pd

    tags = all_tags()
    assert isinstance(tags, list)
    assert len(tags) > 0
    assert sorted(tags) == sorted(OBJECT_TAG_REGISTER)

    for tag in tags:
        assert isinstance(tag, tuple)
        assert len(tag) == 4
        assert isinstance(tag[0], str)
        assert isinstance(tag[1], str)
        assert isinstance(tag[2], (str, tuple))
        assert isinstance(tag[3], str)

    # return_names=False gives tag names only, without duplicates
    names = all_tags(return_names=False)
    assert sorted(names) == sorted(OBJECT_TAG_LIST)
    assert len(names) == len(set(names))

    # as_dataframe takes precedence and names its columns
    frame = all_tags(as_dataframe=True)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["name", "scitype", "type", "description"]
    assert len(frame) == len(OBJECT_TAG_REGISTER)


def test_all_tags_filters_by_parent_type():
    """``parent_types`` subsets the register to that parent type."""
    result = all_tags(parent_types="scaler_strategy", return_names=False)

    assert "fit_per_sequence" in result
    assert "is_label_encoder" in result
    assert len(result) < len(OBJECT_TAG_LIST)


@pytest.mark.parametrize(
    "filter_tags, expected_in_message",
    [
        ({"not_a_real_tag": True}, "not_a_real_tag"),
        ({"capability:exogenus": True}, "capability:exogenous"),
        ("not_a_real_tag", "not_a_real_tag"),
    ],
)
def test_all_objects_rejects_unknown_filter_tag(filter_tags, expected_in_message):
    """A misspelt tag name is an error, not an empty result.

    The second case also covers the suggestion: the message should name the tag
    the caller probably meant.
    """
    with pytest.raises(KeyError) as excinfo:
        all_objects(filter_tags=filter_tags)
    assert expected_in_message in str(excinfo.value)


def test_all_objects_accepts_known_filter_tags():
    """Validation does not break the normal path."""
    result = all_objects(filter_tags={"object_type": "metric"}, return_names=False)
    assert len(result) > 0


def test_tag_classes_are_not_returned_by_all_objects():
    """Tag classes inherit ``_BaseObject`` and must not pollute the object lookup."""
    found = all_objects(return_names=False)
    assert not any(
        getattr(obj, "get_class_tags", lambda: {})().get("object_type") == "tag"
        for obj in found
    )
