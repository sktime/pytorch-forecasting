from urllib.error import ContentTooShortError, URLError

import pytest

from pytorch_forecasting.data import examples


def test_get_data_by_filename_uses_cached_file(monkeypatch, tmp_path):
    fname = "cached.txt"
    cached_file = tmp_path / fname
    cached_file.write_bytes(b"cached data")
    monkeypatch.setattr(examples, "DATA_PATH", tmp_path)

    def fail_urlretrieve(*args, **kwargs):
        raise AssertionError("urlretrieve should not be called for cached files")

    monkeypatch.setattr(examples, "urlretrieve", fail_urlretrieve)

    result = examples._get_data_by_filename(fname)

    assert result == cached_file
    assert cached_file.read_bytes() == b"cached data"


def test_get_data_by_filename_publishes_successful_download(monkeypatch, tmp_path):
    fname = "downloaded.txt"
    final_file = tmp_path / fname
    calls = []
    monkeypatch.setattr(examples, "DATA_PATH", tmp_path)

    def fake_urlretrieve(url, filename):
        calls.append((url, filename))
        assert filename != final_file
        filename.write_bytes(b"complete data")
        return filename, None

    monkeypatch.setattr(examples, "urlretrieve", fake_urlretrieve)

    result = examples._get_data_by_filename(fname)

    assert result == final_file
    assert final_file.read_bytes() == b"complete data"
    assert len(calls) == 1
    url, download_path = calls[0]
    assert url == examples.BASE_URL + fname
    assert download_path != final_file
    assert not download_path.exists()
    assert list(tmp_path.iterdir()) == [final_file]


@pytest.mark.parametrize(
    "error",
    [
        ContentTooShortError("partial download", b"partial"),
        URLError("network unavailable"),
    ],
)
def test_get_data_by_filename_discards_partial_download(monkeypatch, tmp_path, error):
    fname = "partial.txt"
    final_file = tmp_path / fname
    calls = 0
    monkeypatch.setattr(examples, "DATA_PATH", tmp_path)

    def fake_urlretrieve(url, filename):
        nonlocal calls
        calls += 1
        filename.write_bytes(b"partial")
        raise error

    monkeypatch.setattr(examples, "urlretrieve", fake_urlretrieve)

    with pytest.raises(type(error)) as exc_info:
        examples._get_data_by_filename(fname)

    assert exc_info.value is error
    assert not final_file.exists()
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(type(error)):
        examples._get_data_by_filename(fname)

    assert calls == 2
