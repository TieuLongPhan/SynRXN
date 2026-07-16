"""Typed access to the packaged SynRXN dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


class Task(str, Enum):
    """Supported SynRXN benchmark families."""

    AAM = "aam"
    CLASSIFICATION = "classification"
    PROPERTY = "property"
    RBL = "rbl"
    SYNTHESIS = "synthesis"

    @classmethod
    def normalize(cls, value: str) -> "Task":
        aliases = {"class": "classification", "prop": "property", "syn": "synthesis"}
        cleaned = str(value).lower().strip("/\\ ")
        try:
            return cls(aliases.get(cleaned, cleaned))
        except ValueError as exc:
            supported = ", ".join(task.value for task in cls)
            raise ValueError(f"unsupported task {value!r}; choose one of: {supported}") from exc


@dataclass(frozen=True)
class DatasetId:
    """Canonical task/name identifier for one dataset."""

    task: Task
    name: str

    @property
    def key(self) -> str:
        return f"{self.task.value}/{self.name}.csv.gz"


@dataclass(frozen=True)
class DatasetMetadata:
    """Catalog description for one dataset artifact."""

    dataset_id: DatasetId
    title: str
    description: str
    benchmark_role: str
    license: str
    citations: tuple[str, ...]
    targets: tuple[str, ...]
    split_values: tuple[str, ...]
    row_identifier: str
    row_identifier_unique: bool

    @property
    def task(self) -> Task:
        return self.dataset_id.task

    @property
    def name(self) -> str:
        return self.dataset_id.name

    @property
    def has_split(self) -> bool:
        return bool(self.split_values)


class DatasetCatalog:
    """Load and query the authoritative dataset metadata registry."""

    def __init__(self, metadata_path: Optional[Path] = None) -> None:
        if metadata_path is None:
            resource = files("synrxn.data").joinpath("metadata.yaml")
            try:
                text = resource.read_text(encoding="utf8")
            except FileNotFoundError:
                # Editable/source checkout: the canonical catalog lives in Data/.
                checkout_catalog = Path(__file__).resolve().parents[2] / "Data" / "metadata.yaml"
                text = checkout_catalog.read_text(encoding="utf8")
        else:
            text = Path(metadata_path).read_text(encoding="utf8")
        raw = yaml.safe_load(text) or {}
        if raw.get("schema_version") != "1.0":
            raise ValueError(f"unsupported catalog schema: {raw.get('schema_version')!r}")
        records = raw.get("datasets")
        if not isinstance(records, dict) or not records:
            raise ValueError("catalog.datasets must be a non-empty mapping")
        self.schema_version = raw["schema_version"]
        self.column_definitions: dict[str, dict[str, Any]] = raw.get(
            "column_definitions", {}
        )
        self._datasets = {
            self._metadata_from_record(key, value) for key, value in records.items()
        }
        self._by_id = {item.dataset_id: item for item in self._datasets}

    @staticmethod
    def _metadata_from_record(key: str, value: dict[str, Any]) -> DatasetMetadata:
        path = Path(key)
        if len(path.parts) != 2:
            raise ValueError(f"invalid catalog dataset key: {key!r}")
        filename = path.name
        if filename.endswith(".csv.gz"):
            name = filename[: -len(".csv.gz")]
        elif filename.endswith(".csv"):
            name = filename[: -len(".csv")]
        else:
            raise ValueError(f"unsupported catalog artifact extension: {key!r}")
        dataset_id = DatasetId(Task.normalize(path.parts[0]), name)
        return DatasetMetadata(
            dataset_id=dataset_id,
            title=str(value["title"]),
            description=str(value["description"]),
            benchmark_role=str(value["benchmark_role"]),
            license=str(value["license"]),
            citations=tuple(value.get("citations") or ()),
            targets=tuple(value.get("targets") or ()),
            split_values=tuple(str(item) for item in value.get("split_values") or ()),
            row_identifier=str(value.get("row_identifier") or "r_id"),
            row_identifier_unique=bool(value.get("row_identifier_unique", False)),
        )

    def list(
        self,
        task: Optional[str | Task] = None,
        has_split: Optional[bool] = None,
    ) -> list[DatasetMetadata]:
        """Return datasets filtered by task and published-split availability."""
        normalized_task = None
        if task is not None:
            normalized_task = task if isinstance(task, Task) else Task.normalize(task)
        result: Iterable[DatasetMetadata] = self._datasets
        if normalized_task is not None:
            result = (item for item in result if item.task == normalized_task)
        if has_split is not None:
            result = (item for item in result if item.has_split is has_split)
        return sorted(result, key=lambda item: (item.task.value, item.name))

    def get(self, task: str | Task, name: str) -> DatasetMetadata:
        """Return one dataset or raise a helpful ``KeyError``."""
        normalized_task = task if isinstance(task, Task) else Task.normalize(task)
        normalized_name = str(name).strip()
        for suffix in (".csv.gz", ".csv"):
            if normalized_name.lower().endswith(suffix):
                normalized_name = normalized_name[: -len(suffix)]
                break
        dataset_id = DatasetId(normalized_task, normalized_name)
        try:
            return self._by_id[dataset_id]
        except KeyError as exc:
            available = ", ".join(item.name for item in self.list(normalized_task))
            raise KeyError(
                f"unknown dataset {normalized_task.value}/{normalized_name}; "
                f"available: {available}"
            ) from exc

    def available_names(self, task: str | Task) -> list[str]:
        return [item.name for item in self.list(task)]
