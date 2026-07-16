"""Verify SynRXN release artifacts against ``manifest.json``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

CHUNK = 1024 * 1024
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
DATA_TASKS = {"aam", "classification", "property", "rbl", "synthesis"}


class ManifestContractError(ValueError):
    """Raised when a manifest does not satisfy the SynRXN contract."""


def sha256_of_path(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate the manifest envelope and return its artifact entries."""
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ManifestContractError(
            f"unsupported or missing schema_version {schema_version!r}; "
            f"supported: {supported}"
        )

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ManifestContractError("manifest.dataset must be an object")

    files = dataset.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestContractError("manifest.dataset.files must be a non-empty array")

    seen = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ManifestContractError(f"dataset.files[{index}] must be an object")
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ManifestContractError(f"dataset.files[{index}].key must be non-empty")
        pure = PurePosixPath(key)
        if pure.is_absolute() or ".." in pure.parts:
            raise ManifestContractError(f"unsafe artifact key: {key!r}")
        if key in seen:
            raise ManifestContractError(f"duplicate artifact key: {key!r}")
        seen.add(key)

        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ManifestContractError(f"invalid size for {key!r}: {size!r}")
        sha256 = entry.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in sha256)
        ):
            raise ManifestContractError(f"invalid sha256 for {key!r}")
    return files


def candidate_roots(manifest_path: Path) -> List[Path]:
    """Return likely roots for artifact keys such as ``aam/ecoli.csv.gz``."""
    candidates = [manifest_path.resolve().parent / "Data", Path.cwd() / "Data"]
    roots: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return roots


def discovered_artifact_keys(root: Path) -> set[str]:
    """Discover release data files below the five supported task directories."""
    keys = set()
    for task in DATA_TASKS:
        task_root = root / task
        if not task_root.is_dir():
            continue
        for path in task_root.rglob("*"):
            if path.is_file():
                keys.add(path.relative_to(root).as_posix())
    return keys


def verify_with_root(
    manifest: Dict[str, Any], root: Path, allow_unexpected: bool = False
) -> Dict[str, Any]:
    """Verify every declared artifact against *root*."""
    files = manifest_files(manifest)
    missing = []
    size_mismatch = []
    checksum_mismatch = []

    for entry in files:
        key = entry["key"]
        path = root.joinpath(*PurePosixPath(key).parts)
        if not path.is_file():
            missing.append(key)
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size"]:
            size_mismatch.append((key, entry["size"], actual_size))
        actual_checksum = sha256_of_path(path)
        if actual_checksum.lower() != entry["sha256"].lower():
            checksum_mismatch.append((key, entry["sha256"], actual_checksum))

    expected = {entry["key"] for entry in files}
    unexpected = [] if allow_unexpected else sorted(discovered_artifact_keys(root) - expected)
    return {
        "root": str(root),
        "missing": missing,
        "unexpected": unexpected,
        "size_mismatch": size_mismatch,
        "checksum_mismatch": checksum_mismatch,
        "files_checked": len(files),
    }


def verification_succeeded(result: Dict[str, Any]) -> bool:
    """Return whether a verification result is a complete success."""
    return result["files_checked"] > 0 and not any(
        result[name]
        for name in ("missing", "unexpected", "size_mismatch", "checksum_mismatch")
    )


def print_result(result: Dict[str, Any], quiet: bool = False) -> None:
    """Print a concise verification summary and actionable mismatch details."""
    if quiet and verification_succeeded(result):
        return
    print(f"Root: {result['root']}")
    print(
        f"  checked={result['files_checked']} missing={len(result['missing'])} "
        f"unexpected={len(result['unexpected'])} "
        f"size_mismatch={len(result['size_mismatch'])} "
        f"checksum_mismatch={len(result['checksum_mismatch'])}"
    )
    for name in ("missing", "unexpected", "size_mismatch", "checksum_mismatch"):
        if result[name]:
            print(f"  {name}: {result[name][:10]}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "-m", default="manifest.json")
    parser.add_argument(
        "--root",
        type=Path,
        help="Artifact root containing aam/, classification/, property/, rbl/, synthesis/",
    )
    parser.add_argument(
        "--allow-unexpected",
        action="store_true",
        help="Do not fail when the data root contains artifacts absent from the manifest",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        manifest_files(manifest)
    except (OSError, json.JSONDecodeError, ManifestContractError) as exc:
        print(f"invalid manifest: {exc}", file=sys.stderr)
        return 2

    roots = [args.root.resolve()] if args.root else candidate_roots(manifest_path)
    best_result = None
    for root in roots:
        result = verify_with_root(
            manifest, root, allow_unexpected=bool(args.allow_unexpected)
        )
        if best_result is None or len(result["missing"]) < len(best_result["missing"]):
            best_result = result
        if verification_succeeded(result):
            print_result(result, quiet=args.quiet)
            if not args.quiet:
                print("All declared artifacts verified successfully.")
            return 0

    if best_result is not None:
        print_result(best_result, quiet=False)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
