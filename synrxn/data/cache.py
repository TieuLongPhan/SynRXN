"""Version-safe, atomic artifact cache helpers."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _safe_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._") or "unknown"


@dataclass(frozen=True)
class CacheManager:
    root: Path

    def __post_init__(self) -> None:
        resolved = Path(self.root).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "root", resolved)

    def artifact_path(
        self, source: str, version: object, task: str, name: str, suffix: str = ".csv.gz"
    ) -> Path:
        return (
            self.root
            / "artifacts"
            / _safe_component(source)
            / _safe_component(version)
            / _safe_component(task)
            / f"{_safe_component(name)}{suffix}"
        )

    def write_atomic(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
