"""Expansion readers preserve storage aliases, license boundaries and task inputs."""
import json

import pytest

from synrxn.reconstruction.expansion.catalog import ExpansionCatalog
from synrxn.reconstruction.paths import REPO


def test_checkout_catalog_retains_logical_root_argument():
    catalog = ExpansionCatalog(REPO / "Data/expansion")
    assert catalog.root == REPO / "Data/metadata/expansion"
    assert len(catalog.list()) == 16


@pytest.mark.parametrize("partition,license_spdx,path,error", [
    ("main", "CC-BY-SA-4.0", "main/data.parquet", "ShareAlike"),
    ("main", "CC-BY-4.0", "sharealike/data.parquet", "partition disagrees"),
    ("main", "CC-BY-4.0", "main/../../outside.parquet", "escapes"),
])
def test_catalog_rejects_inconsistent_rights_and_escaped_artifacts(
        tmp_path, partition, license_spdx, path, error):
    metadata = {"schema": "synrxn.expansion.v1",
                "sources": [{"id": "S1", "partition": partition, "license_spdx": license_spdx}],
                "tables": [{"key": "tiny", "source_id": "S1", "path": path}]}
    (tmp_path / "catalog.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        ExpansionCatalog(tmp_path)


def test_task_view_excludes_native_labels_and_skips_unlabelled_rows(tmp_path, monkeypatch):
    metadata = {"schema": "synrxn.expansion.v1",
                "sources": [{"id": "S1", "partition": "main", "license_spdx": "CC-BY-4.0"}],
                "tables": [{"key": "tiny", "source_id": "S1", "path": "main/data.parquet",
                            "task_fields": {"yield": {"inputs": ["reaction"], "targets": ["yield"]}}}]}
    (tmp_path / "catalog.json").write_text(json.dumps(metadata))
    catalog = ExpansionCatalog(tmp_path)
    row = {"record_id": "r1", "group_id": "g1", "split": "test", "evidence": "measured",
           "inputs": {"reaction": "C>>CO", "yield_hint": 99}, "targets": {"yield": 12.5},
           "native": {"label": 12.5}, "context": {"future": "hidden"}}
    monkeypatch.setattr(catalog, "records", lambda *args, **kwargs: iter([row, {**row, "targets": {}}]))
    assert list(catalog.task_view("tiny", "yield")) == [
        {"record_id": "r1", "group_id": "g1", "split": "test", "evidence": "measured",
         "inputs": {"reaction": "C>>CO"}, "targets": {"yield": 12.5}}]
    with pytest.raises(ValueError, match="no implemented view"):
        list(catalog.task_view("tiny", "unknown"))
