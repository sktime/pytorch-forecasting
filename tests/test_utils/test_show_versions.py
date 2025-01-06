"""Tests for the show_versions utility."""

import pathlib
import uuid

from pytorch_forecasting.utils._maint._show_versions import (
    DEFAULT_DEPS_TO_SHOW,
    _get_deps_info,
    show_versions,
)


def test_show_versions_runs():
    """Test that show_versions runs without exceptions."""
    # only prints, should return None
    assert show_versions() is None


def test_show_versions_import_loc():
    """Test that show_version can be imported from root."""
    from pytorch_forecasting import show_versions as show_versions_imported

    assert show_versions == show_versions_imported


def test_deps_info():
    """Test that _get_deps_info returns package/version dict as per contract."""
    deps_info = _get_deps_info()
    assert isinstance(deps_info, dict)
    assert set(deps_info.keys()) == {"pytorch-forecasting"}

    deps_info_default = _get_deps_info(DEFAULT_DEPS_TO_SHOW)
    assert isinstance(deps_info_default, dict)
    assert set(deps_info_default.keys()) == set(DEFAULT_DEPS_TO_SHOW)


def test_deps_info_deps_missing_package_present_directory():
    """Test that _get_deps_info does not fail if a dependency is missing."""
    dummy_package_name = uuid.uuid4().hex

    dummy_folder_path = pathlib.Path(dummy_package_name)
    dummy_folder_path.mkdir()

    assert _get_deps_info([dummy_package_name]) == {dummy_package_name: None}

    dummy_folder_path.rmdir()
