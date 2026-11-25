"""Deterministic Parquet conversion and bounded DuckDB queries for SynRXN."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from itertools import zip_longest
from threading import RLock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

from synrxn.data.catalog import DatasetCatalog, DatasetMetadata, Task
from synrxn.data.release import artifact_path, selected_datasets
from synrxn.reconstruction.paths import physical_path

CONVERTER_VERSION = "2.0"
NUMERIC_TARGETS = {
    "ea",
    "dh",
    "G_act",
    "G_r",
    "lograte",
    "Conversion",
    "yield_pct",
    "temperature_c",
    "time_hours",
    "selectivity",
    "selectivity_value",
    "reported_yield_pct",
    "mass_ion_count",
}


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - clean optional-dependency check
        raise RuntimeError("Parquet support requires: pip install synrxn[query]") from exc
    return pa, pacsv, pq


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DuckDB support requires: pip install synrxn[query]") from exc
    return duckdb


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrow_column_types(columns: Sequence[str]) -> dict[str, Any]:
    """Return the stable logical schema used for a CSV artifact."""
    pa, _, _ = _require_pyarrow()
    return {
        column: pa.int8() if column == "feasible" else pa.float64() if column in NUMERIC_TARGETS else pa.string()
        for column in columns
    }


def _csv_columns(path: Path) -> tuple[str, ...]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf8", errors="replace", newline="") as handle:
        return tuple(next(csv.reader(handle)))


@dataclass(frozen=True)
class ParquetArtifact:
    dataset: str
    path: Path
    rows: int
    source_sha256: str
    parquet_sha256: str
    columns: tuple[str, ...]


def _csv_reader(path: Path, columns: Sequence[str], block_size: int = 8 * 1024 * 1024):
    _, pacsv, _ = _require_pyarrow()
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(use_threads=False, block_size=block_size),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(column_types=arrow_column_types(columns), strings_can_be_null=True),
    )


def _fixed_tables(batches, batch_rows: int = 65536):
    """Align variable-size input batches without retaining a complete dataset."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    pa, _, _ = _require_pyarrow()
    parts, count = [], 0
    for batch in batches:
        offset = 0
        while offset < batch.num_rows:
            take = min(batch_rows - count, batch.num_rows - offset)
            parts.append(batch.slice(offset, take))
            count += take
            offset += take
            if count == batch_rows:
                yield pa.Table.from_batches(parts).combine_chunks()
                parts, count = [], 0
    if parts:
        yield pa.Table.from_batches(parts).combine_chunks()


def _csv_parquet_equal(source: Path, parquet: Path, columns: Sequence[str]) -> tuple[bool, int]:
    """Compare every typed value across independent CSV/Parquet batch boundaries."""
    _, _, pq = _require_pyarrow()
    source_batches = _fixed_tables(_csv_reader(source, columns))
    stored_batches = _fixed_tables(pq.ParquetFile(parquet).iter_batches(batch_size=65536, use_threads=False))
    rows = 0
    for original, stored in zip_longest(source_batches, stored_batches):
        if original is None or stored is None or not original.equals(stored, check_metadata=False):
            return False, rows
        rows += original.num_rows
    return True, rows


def convert_dataset(
    source_path: Path,
    output_path: Path,
    dataset: DatasetMetadata,
    compression: str = "zstd",
    row_group_size: int = 100_000,
) -> ParquetArtifact:
    """Convert one canonical CSV artifact and verify its Parquet round trip."""
    if row_group_size < 1:
        raise ValueError("row_group_size must be positive")
    if not dataset.is_redistributable:
        raise ValueError(f"excluded dataset cannot be exported: {dataset.dataset_id.key}")
    _, _, pq = _require_pyarrow()
    source_path = physical_path(Path(source_path).expanduser().resolve())
    output_path = Path(output_path).resolve()
    columns = _csv_columns(source_path)
    source_sha = sha256_path(source_path)
    reader = _csv_reader(source_path, columns)
    metadata = dict(reader.schema.metadata or {})
    metadata.update(
        {
            b"synrxn.dataset": dataset.dataset_id.key.encode(),
            b"synrxn.source_sha256": source_sha.encode(),
            b"synrxn.converter_version": CONVERTER_VERSION.encode(),
        }
    )
    schema = reader.schema.with_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    try:
        with pq.ParquetWriter(
            temporary, schema, compression=compression, use_dictionary=True, write_statistics=True
        ) as writer:
            for table in _fixed_tables(reader, min(row_group_size, 65536)):
                writer.write_table(table.replace_schema_metadata(metadata), row_group_size=row_group_size)
                rows += table.num_rows
        same_values, observed_rows = _csv_parquet_equal(source_path, temporary, columns)
        if (
            not same_values
            or observed_rows != rows
            or not pq.read_schema(temporary).equals(schema, check_metadata=True)
        ):
            raise RuntimeError(f"Parquet round-trip mismatch for {dataset.dataset_id.key}")
        if sha256_path(source_path) != source_sha:
            raise RuntimeError("CSV source changed during Parquet conversion")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(output_path)
    return ParquetArtifact(
        dataset=f"{dataset.task.value}/{dataset.name}",
        path=output_path,
        rows=rows,
        source_sha256=source_sha,
        parquet_sha256=sha256_path(output_path),
        columns=columns,
    )


def convert_release(
    data_dir: Path,
    output_dir: Path,
    catalog: Optional[DatasetCatalog] = None,
    manifest_path: Optional[Path] = None,
    partition: Optional[str] = None,
) -> list[ParquetArtifact]:
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    catalog = catalog or DatasetCatalog()
    manifest = None
    if manifest_path is not None:
        from synrxn.data.verify_manifest import verify_with_root, verification_succeeded

        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        scope = manifest.get("dataset", {}).get("partition", "main")
        if partition is not None and partition != scope:
            raise ValueError("Parquet and canonical manifest partitions differ")
        partition = scope
        if not verification_succeeded(verify_with_root(manifest, Path(data_dir))):
            raise ValueError("Parquet conversion requires a verified canonical manifest")
    partition = partition or "main"
    members = selected_datasets(catalog, partition)
    if not members:
        raise ValueError(f"no datasets in release partition {partition!r}")
    if manifest:
        declared = {entry["key"] for entry in manifest["dataset"]["files"]}
        missing = {item.dataset_id.key for item in members} - declared
        if missing:
            raise ValueError(f"catalog members absent from manifest: {sorted(missing)}")
    expected_paths = {f"{item.task.value}/{item.name}.parquet" for item in members}
    stale = {p.relative_to(output_dir).as_posix() for p in Path(output_dir).rglob("*.parquet")} - expected_paths
    if stale:
        raise ValueError(f"output directory contains non-member Parquet files: {sorted(stale)}")
    artifacts = []
    for dataset in members:
        source = artifact_path(data_dir, f"{dataset.task.value}/{dataset.name}.csv.gz")
        if not source.is_file():
            source = source.with_suffix("")
        output = output_dir / dataset.task.value / f"{dataset.name}.parquet"
        artifacts.append(convert_dataset(source, output, dataset))
    release = None
    if manifest is not None:
        release = {
            "version": manifest.get("dataset", {}).get("version"),
            "doi": manifest.get("dataset", {}).get("doi"),
            "manifest_sha256": sha256_path(manifest_path),
        }
    index = {
        "schema_version": "1.0",
        "converter_version": CONVERTER_VERSION,
        "partition": partition,
        "release": release,
        "artifacts": [
            {
                "dataset": artifact.dataset,
                "path": artifact.path.relative_to(output_dir).as_posix(),
                "rows": artifact.rows,
                "source_sha256": artifact.source_sha256,
                "parquet_sha256": artifact.parquet_sha256,
                "columns": artifact.columns,
            }
            for artifact in artifacts
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf8")
    return artifacts


def validate_parquet_release(data_dir: Path, parquet_dir: Path, catalog: Optional[DatasetCatalog] = None) -> list[str]:
    """Return validation errors for derived Parquet artifacts."""
    _, _, pq = _require_pyarrow()
    catalog = catalog or DatasetCatalog()
    errors = []
    index = json.loads((Path(parquet_dir) / "index.json").read_text(encoding="utf8"))
    for dataset in selected_datasets(catalog, index.get("partition", "main")):
        source = artifact_path(data_dir, f"{dataset.task.value}/{dataset.name}.csv.gz")
        if not source.is_file():
            source = source.with_suffix("")
        parquet = parquet_dir / dataset.task.value / f"{dataset.name}.parquet"
        if not parquet.is_file():
            errors.append(f"missing parquet: {dataset.task.value}/{dataset.name}")
            continue
        metadata = pq.read_metadata(parquet)
        embedded = metadata.metadata or {}
        if embedded.get(b"synrxn.source_sha256", b"").decode() != sha256_path(source):
            errors.append(f"source checksum mismatch: {dataset.task.value}/{dataset.name}")
        columns = _csv_columns(source)
        same_values, csv_rows = _csv_parquet_equal(source, parquet, columns)
        if not same_values:
            errors.append(f"typed value mismatch: {dataset.task.value}/{dataset.name}")
        if metadata.num_rows != csv_rows:
            errors.append(
                f"row count mismatch: {dataset.task.value}/{dataset.name} "
                f"csv={csv_rows} parquet={metadata.num_rows}"
            )
    return errors


def _indexed_release_errors(release) -> list[str]:
    if release is None:
        return []
    checksum = release.get("manifest_sha256") if isinstance(release, dict) else None
    if not isinstance(checksum, str) or len(checksum) != 64:
        return ["parquet release index has an invalid manifest checksum"]
    return []


def validate_parquet_index(parquet_dir: Path) -> list[str]:
    """Validate a self-contained derived release using its checksum index."""
    parquet_dir = Path(parquet_dir).expanduser().resolve()
    index_path = parquet_dir / "index.json"
    if not index_path.is_file():
        return ["missing parquet release index"]
    try:
        index = json.loads(index_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid parquet release index: {exc}"]
    if index.get("schema_version") != "1.0":
        return [f"unsupported parquet index schema: {index.get('schema_version')!r}"]
    partition = index.get("partition", "main")
    if partition not in {"main", "sharealike"}:
        return [f"unsupported parquet partition: {partition!r}"]
    errors = _indexed_release_errors(index.get("release"))
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return ["parquet release index has no artifacts"]
    for artifact in artifacts:
        dataset_id = artifact.get("dataset", "")
        in_sa = dataset_id.split("/", 1)[0].endswith("_sharealike")
        if in_sa != (partition == "sharealike"):
            errors.append(f"indexed dataset crosses partition: {dataset_id}")
        relative = artifact.get("path", "")
        path = (parquet_dir / relative).resolve()
        try:
            path.relative_to(parquet_dir)
        except ValueError:
            errors.append(f"artifact path escapes release root: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing indexed artifact: {relative}")
        elif sha256_path(path) != artifact.get("parquet_sha256"):
            errors.append(f"parquet checksum mismatch: {relative}")
    indexed_paths = {item.get("path") for item in artifacts}
    for path in parquet_dir.rglob("*.parquet"):
        relative = path.relative_to(parquet_dir).as_posix()
        if relative not in indexed_paths:
            errors.append(f"unindexed parquet artifact: {relative}")
    return errors


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class QueryEngine:
    """Allowlisted read-only queries over dataset-specific Parquet files."""

    def __init__(self, parquet_dir: Path, catalog: Optional[DatasetCatalog] = None):
        self.parquet_dir = Path(parquet_dir).expanduser().resolve()
        self.duckdb = _require_duckdb()
        self.connection = self.duckdb.connect(":memory:")
        self.connection.execute("SET threads = 2")
        self._lock = RLock()
        index_path = self.parquet_dir / "index.json"
        self.index = json.loads(index_path.read_text(encoding="utf8")) if index_path.is_file() else None
        self.partition = (self.index or {}).get("partition", "main")
        self.catalog = catalog or DatasetCatalog()
        if isinstance(self.catalog, DatasetCatalog):
            self.catalog = self.catalog.for_release(self.partition)

    def close(self) -> None:
        self.connection.close()

    def _resolve(self, task: str | Task, name: str) -> tuple[DatasetMetadata, Path, list[str]]:
        dataset = self.catalog.get(task, name)
        if not dataset.is_redistributable:
            raise KeyError(f"excluded dataset is not queryable: {dataset.dataset_id.key}")
        if self.index is not None:
            members = {item["dataset"] for item in self.index.get("artifacts", [])}
            if f"{dataset.task.value}/{dataset.name}" not in members:
                raise KeyError(f"dataset absent from this query release: {dataset.dataset_id.key}")
        path = self.parquet_dir / dataset.task.value / f"{dataset.name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Parquet artifact not found: {path}")
        with self._lock:
            schema = self.connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        return dataset, path, [row[0] for row in schema]

    def query(
        self,
        task: str | Task,
        name: str,
        *,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, object]] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
        source_order: bool = False,
        limit: int = 100,
        offset: int = 0,
    ):
        """Return a pandas DataFrame from a bounded, parameterized query."""
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        dataset, path, available = self._resolve(task, name)
        selected = list(columns) if columns is not None else available
        unknown = set(selected) - set(available)
        if unknown:
            raise KeyError(f"unknown projected columns: {sorted(unknown)}")
        reader = "read_parquet(?, file_row_number=true)" if source_order else "read_parquet(?)"
        sql = f"SELECT {', '.join(_quote_identifier(x) for x in selected)} FROM {reader}"
        params: list[object] = [str(path)]
        clauses = []
        for column, expected in (filters or {}).items():
            if column not in available:
                raise KeyError(f"unknown filter column: {column}")
            if isinstance(expected, (list, tuple, set, frozenset)):
                values = list(expected)
                if not values:
                    clauses.append("FALSE")
                    continue
                clauses.append(f"{_quote_identifier(column)} IN ({', '.join('?' for _ in values)})")
                params.extend(values)
            else:
                clauses.append(f"{_quote_identifier(column)} = ?")
                params.append(expected)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by is not None:
            if order_by not in available:
                raise KeyError(f"unknown order column: {order_by}")
            sql += f" ORDER BY {_quote_identifier(order_by)} {'DESC' if descending else 'ASC'}"
        elif source_order:
            sql += " ORDER BY file_row_number"
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._lock:
            frame = self.connection.execute(sql, params).fetchdf()
        frame.attrs["synrxn"] = {
            "dataset": f"{dataset.task.value}/{dataset.name}",
            "parquet_sha256": self.artifact_checksum(dataset.task, dataset.name),
            "source_sha256": self._source_checksum(dataset),
            "converter_version": (self.index or {}).get("converter_version"),
            "release": (self.index or {}).get("release"),
            "columns": selected,
            "target_units": {name: unit for name, unit in dataset.target_units.items() if name in selected},
            "target_kinds": {name: kind for name, kind in dataset.target_kinds.items() if name in selected},
            "filters": filters or {},
            "order_by": order_by,
            "descending": descending,
            "ordering": order_by or ("source_row" if source_order else None),
            "limit": limit,
            "offset": offset,
        }
        return frame

    def _source_checksum(self, dataset: DatasetMetadata) -> Optional[str]:
        key = f"{dataset.task.value}/{dataset.name}"
        for artifact in (self.index or {}).get("artifacts", []):
            if artifact.get("dataset") == key:
                return artifact.get("source_sha256")
        return None

    def artifact_checksum(self, task: str | Task, name: str) -> str:
        """Return the indexed derived checksum, hashing only without an index."""
        dataset = self.catalog.get(task, name)
        key = f"{dataset.task.value}/{dataset.name}"
        for artifact in (self.index or {}).get("artifacts", []):
            if artifact.get("dataset") == key and artifact.get("parquet_sha256"):
                return str(artifact["parquet_sha256"])
        path = self.parquet_dir / dataset.task.value / f"{dataset.name}.parquet"
        return sha256_path(path)

    def iter_batches(self, task: str | Task, name: str, batch_size: int = 50_000) -> Iterator[Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        _, path, _ = self._resolve(task, name)
        with self._lock:
            reader = self.connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).to_arrow_reader(
                batch_size=batch_size
            )
            yield from reader

    def stats(self, task: str | Task, name: str) -> dict[str, Any]:
        """Return bounded, allowlisted statistics for catalog targets and splits."""
        dataset, path, available = self._resolve(task, name)
        with self._lock:
            row_count = self.connection.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
            split_counts = {}
            split_column = dataset.split_column
            if split_column in available:
                quoted_split = _quote_identifier(split_column)
                split_counts = dict(
                    self.connection.execute(
                        f"SELECT {quoted_split}, count(*) FROM read_parquet(?) "
                        f"WHERE {quoted_split} IS NOT NULL "
                        f"GROUP BY {quoted_split} ORDER BY {quoted_split}",
                        [str(path)],
                    ).fetchall()
                )
            targets = {}
            for target in dataset.targets:
                if target not in available:
                    continue
                quoted = _quote_identifier(target)
                if target in NUMERIC_TARGETS:
                    count, minimum, maximum, average = self.connection.execute(
                        f"SELECT count({quoted}), min({quoted}), max({quoted}), avg({quoted}) " "FROM read_parquet(?)",
                        [str(path)],
                    ).fetchone()
                    targets[target] = {
                        "kind": "numeric",
                        "count": count,
                        "min": minimum,
                        "max": maximum,
                        "mean": average,
                    }
                else:
                    distinct = self.connection.execute(
                        f"SELECT count(DISTINCT {quoted}) FROM read_parquet(?)",
                        [str(path)],
                    ).fetchone()[0]
                    top = self.connection.execute(
                        f"SELECT {quoted}, count(*) AS n FROM read_parquet(?) "
                        f"WHERE {quoted} IS NOT NULL GROUP BY {quoted} ORDER BY n DESC LIMIT 8",
                        [str(path)],
                    ).fetchall()
                    targets[target] = {
                        "kind": "categorical",
                        "unique": distinct,
                        "top": [{"value": value, "count": count} for value, count in top],
                    }
        return {
            "rows": row_count,
            "split_counts": split_counts,
            "targets": targets,
            "split_column": split_column,
            "split_status": dataset.split_status if split_counts else None,
        }


class DatasetScan:
    """Lazy, dataset-bound facade over :class:`QueryEngine`."""

    def __init__(
        self,
        parquet_dir: Path,
        task: str | Task,
        name: str,
        catalog: Optional[DatasetCatalog] = None,
    ) -> None:
        self.engine = QueryEngine(parquet_dir, catalog)
        self.task = task
        self.name = name

    def collect(self, **query_options):
        """Execute a bounded query and return a pandas DataFrame."""
        return self.engine.query(self.task, self.name, **query_options)

    def iter_batches(self, batch_size: int = 50_000) -> Iterator[Any]:
        return self.engine.iter_batches(self.task, self.name, batch_size=batch_size)

    def stats(self) -> dict[str, Any]:
        return self.engine.stats(self.task, self.name)

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> "DatasetScan":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data-dir", type=Path, default=Path("Data"))
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    build.add_argument("--partition", choices=("main", "sharealike"))
    verify = sub.add_parser("verify")
    verify.add_argument("--data-dir", type=Path, default=Path("Data"))
    verify.add_argument("--parquet-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        artifacts = convert_release(
            args.data_dir.resolve(),
            args.output_dir.resolve(),
            manifest_path=args.manifest.resolve(),
            partition=args.partition,
        )
        print(f"Parquet release: {len(artifacts)} artifacts -> {args.output_dir}")
        return 0
    errors = validate_parquet_release(args.data_dir.resolve(), args.parquet_dir.resolve())
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Parquet validation: errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
