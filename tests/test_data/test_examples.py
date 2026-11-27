from urllib.error import URLError

import pytest

from pytorch_forecasting.data import examples


def test_get_data_by_filename_download_failure(tmp_path, monkeypatch):
    """Failed download raises a clear error and leaves no partial file."""

    def failing_urlretrieve(url, filename):
        # write part of a file, then fail like a broken connection
        with open(filename, "w") as f:
            f.write("partial")
        raise URLError("connection lost")

    monkeypatch.setattr(examples, "DATA_PATH", tmp_path)
    monkeypatch.setattr(examples, "urlretrieve", failing_urlretrieve)

    with pytest.raises(RuntimeError, match="Failed to download dataset"):
        examples._get_data_by_filename("test.parquet")

    assert list(tmp_path.iterdir()) == []
