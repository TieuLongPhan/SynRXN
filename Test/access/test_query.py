from pathlib import Path

import pytest

from synrxn.data import DataLoader, DatasetCatalog
from synrxn.reconstruction.paths import REPO, physical_path
from synrxn.access.query import (
    DatasetScan,
    QueryEngine,
    CONVERTER_VERSION,
    convert_release,
    validate_parquet_index,
    validate_parquet_release,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")


def test_deterministic_parquet_conversion_and_index(parquet_release):
    output, catalog, artifact = parquet_release
    assert artifact.rows == 273
    assert artifact.columns[0] == "r_id"
    assert validate_parquet_index(output) == []
    assert validate_parquet_release(Path("Data"), output, catalog) == []
    metadata = pq.read_metadata(artifact.path).metadata
    assert metadata[b"synrxn.source_sha256"].decode() == artifact.source_sha256
    assert metadata[b"synrxn.converter_version"] == CONVERTER_VERSION.encode()


def test_release_conversion_accepts_relative_output_directory(tmp_path, monkeypatch, ecoli_catalog):
    monkeypatch.chdir(tmp_path)
    source_root = REPO / "Data"
    artifacts = convert_release(source_root, Path("relative-parquet"), ecoli_catalog)
    assert artifacts[0].path.is_absolute()
    assert validate_parquet_index(Path("relative-parquet")) == []


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
    loader = DataLoader(task="aam", source="local", data_dir="Data", parquet_dir=output)
    table = loader.load("ecoli", columns=["r_id"], nrows=3, format="arrow")
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    scan = loader.scan("ecoli")
    assert isinstance(scan, DatasetScan)
    assert len(scan.collect(columns=["r_id"], limit=2)) == 2
    scan.close()


def test_streaming_conversion_preserves_quoted_newlines_nulls_and_binary_labels(tmp_path, monkeypatch):
    import csv
    import gzip
    import synrxn.access.query as query

    source = tmp_path / "source.csv.gz"
    with gzip.open(source, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "reaction_smiles", "feasible", "additional_note"])
        for index in range(300):
            writer.writerow([str(index), "C>>CO", index % 2, 'a comma, a quote " and\na newline' if index % 3 else ""])
    original_reader = query._csv_reader
    monkeypatch.setattr(query, "_csv_reader", lambda path, columns: original_reader(path, columns, block_size=1024))
    metadata = DatasetCatalog().get("feasibility", "frea_expert")
    artifact = query.convert_dataset(source, tmp_path / "data.parquet", metadata, row_group_size=17)
    stored = pq.read_table(artifact.path)
    assert artifact.rows == 300 and stored.schema.field("feasible").type == pa.int8()
    assert stored.column("additional_note").null_count == 100
    assert stored.column("additional_note")[1].as_py() == 'a comma, a quote " and\na newline'
    assert stored.column("sample_id")[299].as_py() == "299"


def test_roundtrip_validation_detects_changed_values_with_unchanged_row_count(parquet_release):
    output, catalog, artifact = parquet_release
    stored = pq.read_table(artifact.path)
    column = stored.schema.get_field_index("r_id")
    corrupted = stored.set_column(column, "r_id", pa.array(["changed"] * stored.num_rows))
    pq.write_table(corrupted, artifact.path)
    errors = validate_parquet_release(Path("Data"), output, catalog)
    assert any("typed value mismatch" in error for error in errors)


def test_query_provenance_keeps_distinct_units_for_the_same_target_name(tmp_path):
    import csv
    import gzip

    catalog = DatasetCatalog()
    datasets = [catalog.get("property", name) for name in ["b97xd3", "rad6re"]]

    class Subset:
        def list(self):
            return datasets

        def get(self, task, name):
            return catalog.get(task, name)

    sources = tmp_path / "Data"
    for dataset in datasets:
        source = physical_path(Path("Data").resolve() / dataset.dataset_id.key)
        target = sources / dataset.dataset_id.key
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(source, "rt", newline="") as handle, gzip.open(target, "wt", newline="") as dest:
            reader = csv.DictReader(handle)
            writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerow(next(reader))
    output = tmp_path / "Parquet"
    convert_release(sources, output, Subset())
    engine = QueryEngine(output, Subset())
    b97 = engine.query("property", "b97xd3", columns=["dh"], limit=1)
    rad6 = engine.query("property", "rad6re", columns=["dh"], limit=1)
    assert b97.attrs["synrxn"]["target_units"] == {"dh": "kcal/mol"}
    assert rad6.attrs["synrxn"]["target_units"] == {"dh": "eV"}
    assert b97.attrs["synrxn"]["target_kinds"] == {"dh": "reaction_enthalpy"}
    assert b97.iloc[0]["dh"] == pytest.approx(45.5320734228033)
    assert rad6.iloc[0]["dh"] == pytest.approx(1.937744329)
    engine.close()
