"""Dataset source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .catalog import Task


@dataclass(frozen=True)
class LocalSource:
    """Resolve datasets from a local ``Data/``-style directory."""

    data_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())

    def available_names(self, task: str | Task) -> list[str]:
        normalized = task if isinstance(task, Task) else Task.normalize(task)
        task_dir = self.data_dir / normalized.value
        if not task_dir.is_dir():
            return []
        names = set()
        for path in task_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith(".csv.gz"):
                names.add(path.name[: -len(".csv.gz")])
            elif path.name.endswith(".csv"):
                names.add(path.name[: -len(".csv")])
        return sorted(names)

    def resolve(self, task: str | Task, name: str) -> Optional[Path]:
        normalized = task if isinstance(task, Task) else Task.normalize(task)
        clean_name = str(name).strip("/\\ ")
        for suffix in (".csv.gz", ".csv"):
            if clean_name.lower().endswith(suffix):
                clean_name = clean_name[: -len(suffix)]
                break
        for suffix in (".csv.gz", ".csv"):
            path = self.data_dir / normalized.value / f"{clean_name}{suffix}"
            if path.is_file():
                return path
        return None
