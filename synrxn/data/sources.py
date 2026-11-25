"""Dataset source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .catalog import DatasetCatalog, Task
from .release import artifact_keys, artifact_path


@dataclass(frozen=True)
class LocalSource:
    """Resolve datasets from a local ``Data/``-style directory."""

    data_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())

    def available_names(self, task: str | Task) -> list[str]:
        normalized = task if isinstance(task, Task) else Task.normalize(task)
        names = set()
        for key in artifact_keys(self.data_dir, normalized.value):
            parts = Path(key).parts
            if len(parts) == 2 and parts[1].endswith((".csv.gz", ".csv")):
                names.add(parts[1].removesuffix(".gz").removesuffix(".csv"))
        metadata = artifact_path(self.data_dir, "metadata.yaml")
        if (self.data_dir / "families/index.json").is_file() and metadata.is_file():
            for item in DatasetCatalog(metadata).list(normalized):
                if (item.alias_of or not item.is_redistributable) and self.resolve(normalized, item.name):
                    names.add(item.name)
        return sorted(names)

    def resolve(self, task: str | Task, name: str) -> Optional[Path]:
        normalized = task if isinstance(task, Task) else Task.normalize(task)
        clean_name = str(name).strip("/\\ ")
        for suffix in (".csv.gz", ".csv"):
            if clean_name.lower().endswith(suffix):
                clean_name = clean_name[: -len(suffix)]
                break
        for suffix in (".csv.gz", ".csv"):
            path = artifact_path(self.data_dir, f"{normalized.value}/{clean_name}{suffix}")
            if path.is_file():
                return path
        return None
