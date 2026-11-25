"""Typed access to the packaged SynRXN dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from ..reconstruction.paths import physical_path


class Task(str, Enum):
    """Supported SynRXN benchmark families.

    Only families that distribute at least one catalog artifact are listed.
    Feasibility separates expert judgments from generated supervision.
    Mechanism records retain curated annotation and verification tiers.
    Impurity remains audit-only and has no distributable catalog table.
    """

    AAM = "aam"
    CLASSIFICATION = "classification"
    CONDITION = "condition"
    CONDITION_SHAREALIKE = "condition_sharealike"
    PROPERTY = "property"
    RBL = "rbl"
    SELECTIVITY = "selectivity"
    SYNTHESIS = "synthesis"
    YIELD = "yield"
    FEASIBILITY = "feasibility"
    MECHANISM = "mechanism"

    @classmethod
    def normalize(cls, value: str) -> "Task":
        aliases = {
            "class": "classification",
            "cond": "condition",
            "cond_sa": "condition_sharealike",
            "prop": "property",
            "sel": "selectivity",
            "syn": "synthesis",
            "yld": "yield",
            "feas": "feasibility",
            "mech": "mechanism",
        }
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
    license_spdx: str
    license_url: str
    redistribution_partition: str
    citations: tuple[str, ...]
    targets: tuple[str, ...]
    split_values: tuple[str, ...]
    row_identifier: str
    row_identifier_unique: bool
    alias_of: Optional[str] = None
    split_column: str = "split"
    split_status: str = "published"
    target_units: dict[str, str] = field(default_factory=dict, hash=False)
    target_kinds: dict[str, str] = field(default_factory=dict, hash=False)

    @property
    def task(self) -> Task:
        return self.dataset_id.task

    @property
    def name(self) -> str:
        return self.dataset_id.name

    @property
    def has_split(self) -> bool:
        return bool(self.split_values) and self.split_status == "published"

    @property
    def is_sharealike(self) -> bool:
        """Whether redistribution carries a ShareAlike obligation."""
        return self.redistribution_partition == "sharealike"

    @property
    def is_redistributable(self) -> bool:
        """Whether the artifact may be published in a SynRXN release."""
        return self.redistribution_partition in {"main", "sharealike"}


class DatasetCatalog:
    """Load and query the authoritative dataset metadata registry."""

    def __init__(self, metadata_path: Optional[Path] = None) -> None:
        if metadata_path is None:
            resource = files("synrxn.data").joinpath("metadata.yaml")
            try:
                text = resource.read_text(encoding="utf8")
            except FileNotFoundError:
                # Editable/source checkout: the canonical catalog lives in Data/.
                checkout_catalog = Path(__file__).resolve().parents[2] / "Data/metadata/catalog.yaml"
                text = checkout_catalog.read_text(encoding="utf8")
        else:
            text = physical_path(Path(metadata_path).expanduser().resolve()).read_text(encoding="utf8")
        raw = yaml.safe_load(text) or {}
        if raw.get("schema_version") != "1.0":
            raise ValueError(f"unsupported catalog schema: {raw.get('schema_version')!r}")
        records = raw.get("datasets")
        if not isinstance(records, dict) or not records:
            raise ValueError("catalog.datasets must be a non-empty mapping")
        self.schema_version = raw["schema_version"]
        self.column_definitions: dict[str, dict[str, Any]] = raw.get("column_definitions", {})
        self._datasets = {self._metadata_from_record(key, value) for key, value in records.items()}
        self._by_id = {item.dataset_id: item for item in self._datasets}
        self._validate_partitions()

    def _validate_partitions(self) -> None:
        """Fail loudly if license partitions and directories disagree.

        This project's release policy keeps main and ShareAlike artifacts in
        separate directories so each export carries its declared terms.
        """
        from .release import release_keys

        release_keys(
            {
                item.dataset_id.key: {
                    "redistribution_partition": item.redistribution_partition,
                    "license_spdx": item.license_spdx,
                    "alias_of": item.alias_of,
                }
                for item in self._datasets
            }
        )
        valid = {"main", "sharealike", "excluded"}
        for item in sorted(self._datasets, key=lambda d: d.dataset_id.key):
            key = item.dataset_id.key
            partition = item.redistribution_partition
            if partition not in valid:
                raise ValueError(
                    f"{key}: unknown redistribution_partition {partition!r}; "
                    f"choose one of: {', '.join(sorted(valid))}"
                )
            in_sa_dir = item.task.value.endswith("_sharealike")
            if in_sa_dir != (partition == "sharealike"):
                raise ValueError(
                    f"{key}: ShareAlike partition mismatch; directory says "
                    f"{'sharealike' if in_sa_dir else 'main'} but metadata says "
                    f"{partition!r}. ShareAlike artifacts must live in a "
                    "*_sharealike directory and nowhere else."
                )
            if (item.license_spdx == "CC-BY-SA-4.0") != (partition == "sharealike"):
                raise ValueError(
                    f"{key}: license {item.license_spdx!r} is inconsistent with "
                    f"redistribution_partition {partition!r}."
                )

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
            license_spdx=str(value["license_spdx"]),
            license_url=str(value.get("license_url") or ""),
            redistribution_partition=str(value["redistribution_partition"]),
            citations=tuple(value.get("citations") or ()),
            targets=tuple(value.get("targets") or ()),
            split_values=tuple(str(item) for item in value.get("split_values") or ()),
            row_identifier=str(value.get("row_identifier") or "r_id"),
            row_identifier_unique=bool(value.get("row_identifier_unique", False)),
            alias_of=value.get("alias_of"),
            split_column=str(value.get("split_column", "split")),
            split_status=str(value.get("split_status", "published")),
            target_units=dict(value.get("target_units") or {}),
            target_kinds=dict(value.get("target_kinds") or {}),
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
            alias = getattr(self, "_aliases", {}).get(dataset_id)
            if alias is not None:
                return alias
            available = ", ".join(item.name for item in self.list(normalized_task))
            raise KeyError(
                f"unknown dataset {normalized_task.value}/{normalized_name}; " f"available: {available}"
            ) from exc

    def available_names(self, task: str | Task) -> list[str]:
        return [item.name for item in self.list(task)]

    def for_release(self, partition: str = "main") -> "DatasetCatalog":
        """Return a catalog view containing exactly one canonical partition."""
        from .release import selected_datasets

        view = object.__new__(DatasetCatalog)
        view.schema_version = self.schema_version
        view.column_definitions = self.column_definitions
        view._datasets = set(selected_datasets(self, partition))
        view._by_id = {item.dataset_id: item for item in view._datasets}
        view._aliases = {}
        canonical = {item.dataset_id.key: item for item in view._datasets}
        for item in self._datasets:
            if item.alias_of in canonical:
                view._aliases[item.dataset_id] = canonical[item.alias_of]
        for dataset_id, item in getattr(self, "_aliases", {}).items():
            if item.dataset_id.key in canonical:
                view._aliases[dataset_id] = canonical[item.dataset_id.key]
        return view
