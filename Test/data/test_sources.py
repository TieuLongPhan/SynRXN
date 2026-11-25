"""Check catalogue source locks, including split-specific and excluded records."""

import json

from synrxn.data import DatasetCatalog
from synrxn.reconstruction.paths import REPO
from synrxn.reconstruction.provenance.source_locks import lock_partition_matches

SOURCES = REPO / "Data" / "metadata" / "sources"
UPSTREAM_FIELDS = (
    "source_url",
    "source_file",
    "source_commit_or_version",
    "source_sha256",
    "retrieved_at",
)


def _locks():
    for path in sorted(SOURCES.glob("*/*.json")):
        if path.name in {"build_report.json", "lock_coverage.json"}:
            continue
        record = json.loads(path.read_text(encoding="utf8"))
        if record.get("dataset_key"):
            yield path, record


def test_catalog_source_lock_coverage_and_partitions():
    catalog = {item.dataset_id.key: item for item in DatasetCatalog().list()}
    locked = set()
    for path, record in _locks():
        key = record["dataset_key"]
        declared = record.get("redistribution_partition")
        if key not in catalog:
            assert declared == "excluded", f"{path}: {key} has no catalogue entry"
            continue
        locked.add(key)
        expected = catalog[key].redistribution_partition
        assert lock_partition_matches(record, expected), f"{path}: {declared!r} != {expected!r}"
    assert locked == catalog.keys(), f"Missing source locks: {sorted(catalog.keys() - locked)}"


def test_retrofit_locks_declare_what_is_still_unverified():
    """A retrofit lock must never look like a fully pinned source."""
    for path, record in _locks():
        if not str(record.get("lock_status", "")).startswith("retrofit"):
            continue
        outstanding = set(record["outstanding_fields"])
        assert outstanding, f"{path}: retrofit lock claims nothing is outstanding"
        for field in UPSTREAM_FIELDS:
            if record.get(field) is None:
                assert field in outstanding, f"{path}: {field} null but not declared"
            else:
                assert field not in outstanding
