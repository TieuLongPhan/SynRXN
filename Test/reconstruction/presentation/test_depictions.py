"""Check that cached website drawings preserve their exact reviewed inputs."""
import copy
import hashlib
import io
import json
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

import pytest

from synrxn.reconstruction.presentation import depictions

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40"><path d="M1 1L3 3"/></svg>'


def example(tmp_path):
    original = tmp_path / "_static/reactions/example.svg"
    original.parent.mkdir(parents=True)
    original.write_bytes(SVG)
    return {"id": "example", "release_eligible": True, "depiction": "_static/reactions/example.svg",
            "reaction_text": "[NH4+].[Cl-]>O>CN"}, original


def test_diagram_caches_exact_inputs_without_changing_the_catalogue_or_fallback(tmp_path, monkeypatch):
    item, original = example(tmp_path)
    before = copy.deepcopy(item)
    requests = []

    def fetch(request, timeout):
        requests.append(request)
        return io.BytesIO(SVG)

    monkeypatch.setattr(depictions, "urlopen", fetch)
    assert depictions.cache_depiction(item, tmp_path) == "rendered"
    assert depictions.cache_depiction(item, tmp_path) == "cached"
    assert len(requests) == 1
    query = parse_qs(urlparse(requests[0].full_url).query)
    assert query["smi"] == [item["reaction_text"]]
    assert query["abbr"] == ["off"]
    diagram = original.with_suffix(".cdk.svg")
    metadata = ET.parse(diagram).getroot().find(f"{{{depictions.SVG}}}metadata")
    record = json.loads(metadata.text)
    assert record["reaction_sha256"] == hashlib.sha256(item["reaction_text"].encode()).hexdigest()
    assert record["engine"] == "CDK Depict"
    assert original.read_bytes() == SVG
    assert item == before


def test_changed_reaction_cannot_leave_a_stale_diagram_after_a_failed_request(tmp_path, monkeypatch):
    item, original = example(tmp_path)
    monkeypatch.setattr(depictions, "urlopen", lambda *args, **kwargs: io.BytesIO(SVG))
    depictions.cache_depiction(item, tmp_path)
    item["reaction_text"] = "CCO>>CC=O"

    def unavailable(*args, **kwargs):
        raise OSError("service unavailable")

    monkeypatch.setattr(depictions, "urlopen", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        depictions.cache_depiction(item, tmp_path)
    assert not original.with_suffix(".cdk.svg").exists()
    assert original.read_bytes() == SVG


@pytest.mark.parametrize("held", [True, False])
def test_withheld_or_missing_reaction_is_never_sent_to_the_renderer(tmp_path, monkeypatch, held):
    item, original = example(tmp_path)
    item.update(release_eligible=not held, reaction_text=item["reaction_text"] if held else "")

    def unexpected(*args, **kwargs):
        pytest.fail("An ineligible example must not be sent to CDK")

    monkeypatch.setattr(depictions, "urlopen", unexpected)
    assert depictions.cache_depiction(item, tmp_path) == "skipped"
    assert not original.with_suffix(".cdk.svg").exists()


@pytest.mark.parametrize("payload", [
    b"<html>service failure</html>",
    b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
])
def test_non_diagram_responses_are_not_saved_as_previews(tmp_path, monkeypatch, payload):
    item, original = example(tmp_path)
    monkeypatch.setattr(depictions, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    with pytest.raises(ValueError):
        depictions.cache_depiction(item, tmp_path)
    assert not original.with_suffix(".cdk.svg").exists()


def test_catalogue_cannot_write_diagrams_outside_static_assets(tmp_path):
    item, _ = example(tmp_path)
    item["depiction"] = "../elsewhere.svg"
    with pytest.raises(ValueError, match="inside"):
        depictions.cache_depiction(item, tmp_path)
