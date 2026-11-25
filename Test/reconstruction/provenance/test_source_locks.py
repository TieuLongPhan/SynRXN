"""Source joins must retain identity and reject ambiguous or changed payloads."""

import pytest

from synrxn.reconstruction.provenance.source_history import load_tool
from synrxn.reconstruction.provenance.source_locks import source_pin_gaps

audit_join = load_tool("audit_mapping_sources").audit_join


def test_native_join_ignores_row_order_and_preserves_every_annotation():
    upstream = [{"id": 23, "reactions": "C>>C", "ground_truth": "map:23"},
                {"id": 11, "reactions": "N>>N", "ground_truth": "map:11"}]
    local = [{"original": "11", "r_id": "local:1", "reactions": "N>>N", "ground_truth": "map:11"},
             {"original": "23", "r_id": "local:2", "reactions": "C>>C", "ground_truth": "changed"}]
    spec = {"dataset_key": "fixture", "local_identifier": "original", "native_identifier": "id", "sha256": "pinned"}
    report, links = audit_join(local, upstream, spec)
    assert not report["verified"]
    assert report["payload_mismatches"] == [{"local_row": 1, "native_identifier": "23", "columns": ["ground_truth"]}]
    assert [row["upstream_row"] for row in links] == [1, 0]
    assert links[0]["relationship"] == "exact_imported_annotation_payload"
    assert links[1]["relationship"] == "discrepant_payload"


def test_duplicate_native_accessions_cannot_make_a_verified_source_join():
    spec = {"dataset_key": "fixture", "local_identifier": "original", "native_identifier": "id", "sha256": "pinned"}
    with pytest.raises(ValueError, match="ambiguous upstream"):
        audit_join([], [{"id": 1}, {"id": "1"}], spec)
    with pytest.raises(ValueError, match="duplicate local"):
        audit_join([{"original": "1"}, {"original": "1"}], [], spec)


def test_source_pin_formats_require_real_hashes_and_do_not_certify_an_alias():
    record = {"source_repository": "https://example.org/repo", "source_commit": "fixed",
              "retrieved_at": "2026-10-13", "files": [{"dataset_id": "native", "sha256": "a" * 64}]}
    assert source_pin_gaps(record) == []
    record["files"][0]["sha256"] = "not-a-digest"
    assert "source_sha256" in source_pin_gaps(record)
    assert source_pin_gaps({"alias_of": "parent"}) == ["canonical_alias_source_not_independently_pinned"]
