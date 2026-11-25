"""One explicit membership policy for canonical and derived release artifacts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..reconstruction.paths import family_storage, logical_relative, physical_path

PARTITIONS = {"main", "sharealike"}
DATA_TASKS = {
    "aam", "classification", "condition", "condition_sharealike", "property",
    "rbl", "selectivity", "synthesis", "yield", "feasibility", "mechanism",
}
POLICY_VERSION = "1.0"


def artifact_path(root: Path, key: str) -> Path:
    """Resolve a stable release key without changing external release layouts."""
    root = Path(root).expanduser().resolve()
    checkout = root.parent if (root / "families/index.json").is_file() else None
    return physical_path(root / key, checkout)


def artifact_key(root: Path, path: Path) -> str:
    """Keep historical release keys when canonical storage moves in a checkout."""
    root, path = Path(root).resolve(), Path(path).resolve()
    checkout = root.parent if (root / "families/index.json").is_file() else None
    try:
        logical_root = Path(logical_relative(root, checkout))
        return Path(logical_relative(path, checkout)).relative_to(logical_root).as_posix()
    except ValueError:
        return path.relative_to(root).as_posix()


def artifact_keys(root: Path, directory: str) -> set[str]:
    """List logical files from a checkout or a downloaded release directory."""
    root = Path(root).resolve()
    keys = set()
    if root.name == "Data" and (root / "families/index.json").is_file():
        prefix = f"Data/{directory}/"
        keys.update(path.removeprefix("Data/") for path in family_storage(root.parent)
                    if path.startswith(prefix) and physical_path(path, root.parent).is_file())
    base = artifact_path(root, directory)
    if base.is_dir():
        keys.update(artifact_key(root, path) for path in base.rglob("*") if path.is_file())
    return keys


def release_keys(
    records: Mapping[str, Mapping[str, Any]],
    partition: str = "main",
    include_aliases: bool = False,
) -> set[str]:
    """Select distributable datasets, rejecting inconsistent declarations.

    Older metadata without a partition is treated as main for compatibility.
    Aliases remain locally loadable but are omitted from canonical exports.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"unsupported release partition: {partition!r}")
    selected = set()
    for key, record in records.items():
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
            raise ValueError(f"unsafe dataset key: {key!r}")
        declared = record.get("redistribution_partition", "main")
        if declared not in PARTITIONS | {"excluded"}:
            raise ValueError(f"{key}: invalid redistribution partition {declared!r}")
        in_sa = path.parts[0].endswith("_sharealike")
        if in_sa != (declared == "sharealike"):
            raise ValueError(f"{key}: ShareAlike directory/partition mismatch")
        license_id = record.get("license_spdx")
        if license_id is not None:
            if (license_id == "CC-BY-SA-4.0") != (declared == "sharealike"):
                raise ValueError(f"{key}: license/partition mismatch")
            if declared != "excluded" and license_id == "LicenseRef-Unknown":
                raise ValueError(f"{key}: unresolved rights must be excluded")
        alias = record.get("alias_of")
        if alias:
            if alias not in records or alias == key or records[alias].get("alias_of"):
                raise ValueError(f"{key}: invalid canonical alias target {alias!r}")
            if records[alias].get("redistribution_partition", "main") != declared:
                raise ValueError(f"{key}: alias crosses release partitions")
        if declared == partition and (include_aliases or not alias):
            selected.add(key)
    return selected


def release_artifact_keys(
    root: Path, records: Mapping[str, Mapping[str, Any]], partition: str
) -> set[str]:
    """Discover files for one partition, never scanning raw or excluded data.

    Unexpected unregistered CSVs are retained so verification detects them.
    Source/audit descriptions are metadata, not normalized reaction artifacts.
    """
    selected = release_keys(records, partition)
    directories = {task for task in DATA_TASKS
                   if task.endswith("_sharealike") == (partition == "sharealike")}
    keys = set()
    for directory in directories:
        for key in artifact_keys(root, directory):
            if PurePosixPath(key).name == "manifest.json":
                continue
            if key in records and key not in selected:
                continue
            keys.add(key)
    if partition == "main":
        base = artifact_path(root, "sources")
        if base.is_dir():
            keys.update(artifact_key(root, p)
                        for p in base.rglob("*") if p.is_file()
                        and p.relative_to(base).parts[0] != "previous")
        try:
            checkout_data = logical_relative(Path(root).resolve()) == "Data"
        except ValueError:
            checkout_data = False
        # The checkout README describes storage organization; external release
        # folders may still supply their own distributable dataset README.
        helpers = ("metadata.yaml", "LICENSE") if checkout_data else ("metadata.yaml", "LICENSE", "README.md")
        keys.update(name for name in helpers
                    if artifact_path(root, name).is_file())
    return keys


def selected_datasets(catalog, partition: str = "main"):
    """Apply the policy to a typed catalog or a compatible small catalog view."""
    datasets = catalog.list()
    records = {
        item.dataset_id.key: {
            "redistribution_partition": item.redistribution_partition,
            "license_spdx": item.license_spdx,
            "alias_of": getattr(item, "alias_of", None),
        }
        for item in datasets
    }
    keys = release_keys(records, partition)
    return [item for item in datasets if item.dataset_id.key in keys]
