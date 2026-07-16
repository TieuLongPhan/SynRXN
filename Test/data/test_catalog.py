import gzip

import pandas as pd
import pytest

from synrxn.data import CacheManager, DataLoader, DatasetCatalog, LocalSource, Task


def test_packaged_catalog_covers_current_inventory():
    catalog = DatasetCatalog()
    assert len(catalog.list()) == 33
    assert catalog.get("prop", "rgd1").targets == ("ea",)
    assert catalog.get(Task.CLASSIFICATION, "ecreact").row_identifier_unique is False
    assert "schneider_b" in catalog.available_names("classification")


def test_catalog_rejects_unknown_task_and_dataset():
    with pytest.raises(ValueError, match="unsupported task"):
        Task.normalize("unknown")
    with pytest.raises(KeyError, match="unknown dataset"):
        DatasetCatalog().get("property", "missing")


def test_local_source_loader_projection_filter_and_batches(tmp_path):
    task_dir = tmp_path / "classification"
    task_dir.mkdir()
    path = task_dir / "schneider_b.csv.gz"
    frame = pd.DataFrame(
        {
            "r_id": ["r1", "r2", "r3"],
            "rxn": ["A>>B", "C>>D", "E>>F"],
            "label": [1, 2, 3],
            "split": ["train", "test", "test"],
        }
    )
    frame.to_csv(path, index=False, compression="gzip")

    source = LocalSource(tmp_path)
    assert source.available_names("class") == ["schneider_b"]
    assert source.resolve("classification", "schneider_b") == path

    loader = DataLoader("class", source="local", data_dir=tmp_path, cache_dir=None)
    loaded = loader.load(
        "schneider_b",
        columns=["r_id", "split"],
        filters={"split": "test"},
    )
    assert loaded.to_dict("records") == [
        {"r_id": "r2", "split": "test"},
        {"r_id": "r3", "split": "test"},
    ]
    assert loader.describe("schneider_b").title == "Schneider balanced classification"
    assert [
        len(batch) for batch in loader.iter_batches("schneider_b", batch_size=2)
    ] == [2, 1]


def test_local_loader_nrows_and_errors(tmp_path):
    task_dir = tmp_path / "property"
    task_dir.mkdir()
    with gzip.open(task_dir / "rgd1.csv.gz", "wt", encoding="utf8") as handle:
        handle.write("r_id,aam,ea,split\n1,A>>B,1.0,train\n2,C>>D,2.0,test\n")
    loader = DataLoader("property", source="local", data_dir=tmp_path, cache_dir=None)
    assert len(loader.load("rgd1", nrows=1)) == 1
    with pytest.raises(KeyError, match="filter column"):
        loader.load("rgd1", columns=["r_id"], filters={"split": "train"})
    with pytest.raises(FileNotFoundError, match="Local dataset"):
        loader.load("missing")


def test_cache_manager_namespaces_and_writes_atomically(tmp_path):
    cache = CacheManager(tmp_path)
    first = cache.artifact_path("github", "v1.1.1", "aam", "sample")
    second = cache.artifact_path("github", "v2.0.0", "aam", "sample")
    assert first != second
    cache.write_atomic(first, b"payload")
    assert first.read_bytes() == b"payload"
    assert not list(first.parent.glob("tmp*"))
