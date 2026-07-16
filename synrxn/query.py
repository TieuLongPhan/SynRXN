"""Deterministic Parquet conversion and bounded DuckDB queries for SynRXN."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from threading import RLock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

from .data.catalog import DatasetCatalog, DatasetMetadata, Task

CONVERTER_VERSION = "1.0"
NUMERIC_TARGETS = {"ea", "dh", "G_act", "G_r", "lograte", "Conversion"}


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
        column: pa.float64() if column in NUMERIC_TARGETS else pa.string()
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


def convert_dataset(
    source_path: Path,
    output_path: Path,
    dataset: DatasetMetadata,
    compression: str = "zstd",
    row_group_size: int = 100_000,
) -> ParquetArtifact:
    """Convert one canonical CSV artifact and verify its Parquet round trip."""
    pa, pacsv, pq = _require_pyarrow()
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    columns = _csv_columns(source_path)
    types = arrow_column_types(columns)
    table = pacsv.read_csv(
        source_path,
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            strings_can_be_null=True,
        ),
    )
    source_sha = sha256_path(source_path)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"synrxn.dataset": dataset.dataset_id.key.encode(),
            b"synrxn.source_sha256": source_sha.encode(),
            b"synrxn.converter_version": CONVERTER_VERSION.encode(),
        }
    )
    table = table.replace_schema_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(
        table,
        temporary,
        compression=compression,
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
    )
    roundtrip = pq.read_table(temporary)
    if not table.equals(roundtrip, check_metadata=True):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet round-trip mismatch for {dataset.dataset_id.key}")
    temporary.replace(output_path)
    return ParquetArtifact(
        dataset=f"{dataset.task.value}/{dataset.name}",
        path=output_path,
        rows=table.num_rows,
        source_sha256=source_sha,
        parquet_sha256=sha256_path(output_path),
        columns=columns,
    )


def convert_release(
    data_dir: Path,
    output_dir: Path,
    catalog: Optional[DatasetCatalog] = None,
    manifest_path: Optional[Path] = None,
) -> list[ParquetArtifact]:
    catalog = catalog or DatasetCatalog()
    artifacts = []
    for dataset in catalog.list():
        source = data_dir / dataset.task.value / f"{dataset.name}.csv.gz"
        if not source.is_file():
            source = source.with_suffix("")
        output = output_dir / dataset.task.value / f"{dataset.name}.parquet"
        artifacts.append(convert_dataset(source, output, dataset))
    release = None
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        release = {
            "version": manifest.get("dataset", {}).get("version"),
            "doi": manifest.get("dataset", {}).get("doi"),
            "manifest_sha256": sha256_path(manifest_path),
        }
    index = {
        "schema_version": "1.0",
        "converter_version": CONVERTER_VERSION,
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


def validate_parquet_release(
    data_dir: Path, parquet_dir: Path, catalog: Optional[DatasetCatalog] = None
) -> list[str]:
    """Return validation errors for derived Parquet artifacts."""
    _, pacsv, pq = _require_pyarrow()
    catalog = catalog or DatasetCatalog()
    errors = []
    for dataset in catalog.list():
        source = data_dir / dataset.task.value / f"{dataset.name}.csv.gz"
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
        csv_rows = pacsv.read_csv(source).num_rows
        if metadata.num_rows != csv_rows:
            errors.append(
                f"row count mismatch: {dataset.task.value}/{dataset.name} "
                f"csv={csv_rows} parquet={metadata.num_rows}"
            )
    return errors


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
    errors = []
    release = index.get("release")
    if release is not None:
        checksum = release.get("manifest_sha256") if isinstance(release, dict) else None
        if not isinstance(checksum, str) or len(checksum) != 64:
            errors.append("parquet release index has an invalid manifest checksum")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return ["parquet release index has no artifacts"]
    for artifact in artifacts:
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
    return errors


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class QueryEngine:
    """Allowlisted read-only queries over dataset-specific Parquet files."""

    def __init__(self, parquet_dir: Path, catalog: Optional[DatasetCatalog] = None):
        self.parquet_dir = Path(parquet_dir).expanduser().resolve()
        self.catalog = catalog or DatasetCatalog()
        self.duckdb = _require_duckdb()
        self.connection = self.duckdb.connect(":memory:")
        self._lock = RLock()
        index_path = self.parquet_dir / "index.json"
        self.index = (
            json.loads(index_path.read_text(encoding="utf8"))
            if index_path.is_file()
            else None
        )

    def close(self) -> None:
        self.connection.close()

    def _resolve(self, task: str | Task, name: str) -> tuple[DatasetMetadata, Path, list[str]]:
        dataset = self.catalog.get(task, name)
        path = self.parquet_dir / dataset.task.value / f"{dataset.name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Parquet artifact not found: {path}")
        with self._lock:
            schema = self.connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
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
                clauses.append(
                    f"{_quote_identifier(column)} IN ({', '.join('?' for _ in values)})"
                )
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

    def iter_batches(
        self, task: str | Task, name: str, batch_size: int = 50_000
    ) -> Iterator[Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        _, path, _ = self._resolve(task, name)
        with self._lock:
            reader = self.connection.execute(
                "SELECT * FROM read_parquet(?)", [str(path)]
            ).to_arrow_reader(batch_size=batch_size)
            yield from reader

    def stats(self, task: str | Task, name: str) -> dict[str, Any]:
        """Return bounded, allowlisted statistics for catalog targets and splits."""
        dataset, path, available = self._resolve(task, name)
        with self._lock:
            row_count = self.connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(path)]
            ).fetchone()[0]
            split_counts = {}
            if "split" in available:
                split_counts = dict(
                    self.connection.execute(
                        'SELECT "split", count(*) FROM read_parquet(?) '
                        'WHERE "split" IS NOT NULL GROUP BY "split" ORDER BY "split"',
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
                        f"SELECT count({quoted}), min({quoted}), max({quoted}), avg({quoted}) "
                        "FROM read_parquet(?)",
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
                        "top": [
                            {"value": value, "count": count} for value, count in top
                        ],
                    }
        return {"rows": row_count, "split_counts": split_counts, "targets": targets}


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
