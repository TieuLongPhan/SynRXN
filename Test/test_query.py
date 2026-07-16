from pathlib import Path

import pytest

from synrxn.data import DataLoader, DatasetCatalog
from synrxn.query import (
    DatasetScan,
    QueryEngine,
    convert_release,
    validate_parquet_index,
    validate_parquet_release,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")


@pytest.fixture()
def parquet_release(tmp_path):
    catalog = DatasetCatalog()
    source_root = Path("Data")
    output = tmp_path / "Parquet"
    one = catalog.get("aam", "ecoli")

    class OneDatasetCatalog:
        def list(self):
            return [one]

        def get(self, task, name):
            return catalog.get(task, name)

    subset = OneDatasetCatalog()
    artifacts = convert_release(source_root, output, subset)
    return output, subset, artifacts[0]


def test_deterministic_parquet_conversion_and_index(parquet_release):
    output, catalog, artifact = parquet_release
    assert artifact.rows == 273
    assert artifact.columns[0] == "r_id"
    assert validate_parquet_index(output) == []
    assert validate_parquet_release(Path("Data"), output, catalog) == []
    metadata = pq.read_metadata(artifact.path).metadata
    assert metadata[b"synrxn.source_sha256"].decode() == artifact.source_sha256
    assert metadata[b"synrxn.converter_version"] == b"1.0"


def test_allowlisted_query_projection_filter_order_and_batches(parquet_release):
    output, catalog, _ = parquet_release
    engine = QueryEngine(output, catalog)
    frame = engine.query(
        "aam",
        "ecoli",
        columns=["r_id", "original_id"],
        source_order=True,
        limit=5,
    )
    assert frame.shape == (5, 2)
    assert frame.attrs["synrxn"]["source_sha256"]
    assert frame.attrs["synrxn"]["ordering"] == "source_row"
    assert sum(batch.num_rows for batch in engine.iter_batches("aam", "ecoli", 100)) == 273
    with pytest.raises(KeyError):
        engine.query("aam", "ecoli", filters={"not_a_column": "x"})
    with pytest.raises(ValueError):
        engine.query("aam", "ecoli", limit=10_001)
    engine.close()


def test_loader_arrow_output_and_lazy_scan(parquet_release):
    output, _, _ = parquet_release
    loader = DataLoader(
        task="aam", source="local", data_dir="Data", parquet_dir=output
    )
    table = loader.load("ecoli", columns=["r_id"], nrows=3, format="arrow")
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    scan = loader.scan("ecoli")
    assert isinstance(scan, DatasetScan)
    assert len(scan.collect(columns=["r_id"], limit=2)) == 2
    scan.close()
