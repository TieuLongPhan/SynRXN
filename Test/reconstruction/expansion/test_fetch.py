"""Cached inputs still require a matching release checksum before use."""
import hashlib

import pytest

from synrxn.reconstruction.expansion import fetch as expansion_fetch


@pytest.mark.parametrize("checksum,expected_status", [("correct", "verified"), ("changed", "unavailable")])
def test_cached_download_is_verified_without_network(tmp_path, monkeypatch, checksum, expected_status):
    monkeypatch.setattr(expansion_fetch, "ROOT", tmp_path)
    monkeypatch.setattr(expansion_fetch, "RAW", tmp_path / "raw")
    target = tmp_path / "raw/S1/tiny.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"id,value\n1,2\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest() if checksum == "correct" else "0" * 64
    monkeypatch.setattr(expansion_fetch.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network"))
    asset = {"name": "tiny.csv", "url": "https://example.org/tiny.csv",
             "bytes": target.stat().st_size, "checksum": "sha256:" + digest}
    result = expansion_fetch.download("S1", asset)
    assert result["status"] == expected_status
    if expected_status == "unavailable":
        assert "checksum differs" in result["error"]
    assert target.read_bytes() == b"id,value\n1,2\n"


def test_download_rejects_source_path_escape_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion_fetch, "RAW", tmp_path / "raw")
    with pytest.raises(ValueError, match="unsafe source path"):
        expansion_fetch.download("S1", {"name": "../../escaped.csv"})
    assert not (tmp_path / "escaped.csv").exists()
