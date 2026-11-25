"""Read verified expansion artifacts and construct target-free model inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from ..paths import physical_path


def group_split(group: str, seed: int = 2027) -> str:
    """Stable 80/10/10 partition; all observations of a parent stay together.

    This is a grouping baseline, not a scaffold, homology or temporal split.
    Source-provided splits remain separate annotations in the native record.
    """
    if not group:
        raise ValueError("a nonempty parent group is required")
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) % 10000
    return "train" if value < 8000 else "validation" if value < 9000 else "test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExpansionCatalog:
    def __init__(self, root: str | Path = "Data/expansion") -> None:
        self.logical_root = Path(root).expanduser().resolve()
        if self.logical_root == Path(__file__).resolve().parents[3] / "Data/metadata/expansion":
            self.logical_root = self.logical_root.parents[1] / "expansion"
        self.root = physical_path(self.logical_root)
        data = json.loads((self.root / "catalog.json").read_text())
        if data.get("schema") != "synrxn.expansion.v1":
            raise ValueError("unsupported expansion catalog schema")
        self.metadata = data
        self.tables = {r["key"]: r for r in data["tables"]}
        self.sources = {r["id"]: r for r in data["sources"]}
        self._validate()

    def _path(self, relative: str) -> Path:
        path = (self.logical_root / relative).resolve()
        if not path.is_relative_to(self.logical_root):
            raise ValueError("artifact escapes expansion root")
        return physical_path(path)

    def _validate(self) -> None:
        for table in self.tables.values():
            source = self.sources[table["source_id"]]
            partition = source["partition"]
            if partition not in {"main", "sharealike"}:
                raise ValueError("excluded source has a distributable artifact")
            if Path(table["path"]).parts[0] != partition:
                raise ValueError("license partition disagrees with artifact path")
            if (source["license_spdx"] == "CC-BY-SA-4.0") != (partition == "sharealike"):
                raise ValueError("ShareAlike obligation is not isolated")
            self._path(table["path"])

    def list(self, source: str | None = None, task: str | None = None) -> list[dict]:
        return [r for r in self.tables.values()
                if (source is None or r["source_id"] == source)
                and (task is None or task in r["tasks"])]

    def records(self, key: str, split: str | None = None) -> Iterator[dict]:
        import pyarrow.parquet as pq

        record = self.tables[key]
        for batch in pq.ParquetFile(self._path(record["path"])).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                if split is not None and row["split"] != split:
                    continue
                for field in ("inputs", "targets", "context", "native"):
                    row[field] = json.loads(row.pop(field + "_json"))
                yield row

    def task_view(self, key: str, task: str, split: str | None = None) -> Iterator[dict]:
        """Only explicitly declared inputs reach a model; native labels stay out.

        Task-specific output fields are selected by the adapter's field mapping.
        Protocols without an implemented field mapping cannot be silently used.
        """
        table = self.tables[key]
        mapping = table.get("task_fields", {}).get(task)
        if mapping is None:
            raise ValueError(f"{key} has no implemented view for {task!r}")
        for row in self.records(key, split=split):
            inputs = {k: row["inputs"][k] for k in mapping["inputs"] if k in row["inputs"]}
            targets = {k: row["targets"][k] for k in mapping["targets"] if k in row["targets"]}
            if not targets:
                continue
            yield {"record_id": row["record_id"], "group_id": row["group_id"],
                   "split": row["split"], "evidence": row["evidence"],
                   "inputs": inputs, "targets": targets}

    def verify(self) -> list[dict]:
        import pyarrow.parquet as pq

        results = []
        for table in self.tables.values():
            path = self._path(table["path"])
            results.append({"key": table["key"],
                            "sha256_match": path.exists() and sha256(path) == table["sha256"],
                            "rows_match": path.exists() and pq.ParquetFile(path).metadata.num_rows == table["rows"]})
        return results

    def native_geometry(self, record: dict, frame: int = 0) -> dict:
        """Read one Transition1x configuration lazily, retaining native units.

        HDF5 paths refer to the separately acquired immutable source file.
        No frame is ever assigned an independent parent/split.
        """
        import h5py

        context = record["context"]
        path = self._path(context["native_asset"])
        with h5py.File(path, "r") as handle:
            group = handle[context["hdf5_group"]]
            result = {}
            for name in group:
                array = group[name]
                if not isinstance(array, h5py.Dataset):
                    continue
                result[name] = (array[frame] if array.ndim and array.shape[0] == context["frames"]
                                else array[()]).tolist()
            return result
