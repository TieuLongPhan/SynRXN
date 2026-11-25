"""Check release versions, candidate labels and built distribution contents."""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import json
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGE_FILES = {
    "synrxn/__init__.py", "synrxn/__main__.py", "synrxn/manifest.schema.json",
    "synrxn/data/metadata.yaml", "synrxn/data/task_protocols.json",
    "synrxn/data/expansion_tasks.json", "synrxn/data/expansion_sources.json",
    "synrxn/reconstruction/provenance/source_manifest.json", "synrxn/reconstruction/provenance/migration.json",
}


def runtime_version(source: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("missing explicit runtime __version__")


def metadata_errors(root: Path, tag: str | None = None) -> tuple[str, list[str]]:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    errors = []
    versions = {"runtime": runtime_version((root / "synrxn/__init__.py").read_text())}
    citation = (root / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation, re.MULTILINE)
    versions["citation"] = match[1] if match else None
    if tag is not None and tag != f"v{version}":
        errors.append(f"release tag {tag!r} differs from v{version}")
    for filename in ("manifest.json", "Data/metadata/sharealike-manifest.json"):
        dataset = json.loads((root / filename).read_text())["dataset"]
        versions[filename] = dataset["version"]
        if dataset.get("release_status") == "candidate" and dataset.get("doi"):
            errors.append(f"{filename}: unpublished candidate claims a DOI")
    for name in ("catalog-data", "catalog-sharealike-data"):
        filename = root / f"doc/_static/{name}.json"
        catalog = json.loads(filename.read_text())
        versions[name] = catalog["release"]["version"]
        script = filename.with_suffix(".js").read_text()
        prefix = "window.SYNRXN_CATALOG = "
        if not script.startswith(prefix) or json.loads(script[len(prefix):]) != catalog:
            errors.append(f"{name}: JSON and JavaScript catalogues disagree")
    for label, actual in versions.items():
        if actual != version:
            errors.append(f"{label}: version {actual!r} differs from {version}")
    if version not in (root / "README.md").read_text() or version not in (root / "doc/whats_new.rst").read_text():
        errors.append("README and release notes must describe the current candidate")
    if re.search(r"(?:rc|a|b|dev)\d+", version):
        if "Unpublished" not in citation:
            errors.append("prerelease citation must state that the artifact is unpublished")
        if re.search(r"^(?:doi|date-released):", citation, re.MULTILINE):
            errors.append("unpublished prerelease citation claims a DOI or release date")
    return version, errors


def distribution_errors(directory: Path, version: str) -> list[str]:
    errors = []
    wheels, sources = sorted(directory.glob("*.whl")), sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        return ["distribution directory must contain exactly one wheel and one source distribution"]
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
        missing = REQUIRED_PACKAGE_FILES - members
        errors.extend(f"wheel missing {name}" for name in sorted(missing))
        metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            errors.append("wheel must contain one METADATA record")
        else:
            metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
            if metadata["Name"] != "synrxn" or metadata["Version"] != version:
                errors.append("wheel distribution name or version differs from pyproject")
        runtime = "synrxn/__init__.py"
        if runtime in members and runtime_version(wheel.read(runtime).decode()) != version:
            errors.append("wheel runtime version differs from package metadata")
        for name in members:
            if name.startswith(("archive/", "Data/", "Test/", "scripts/")) or name.endswith(
                    (".duckdb", ".npz", ".csv.gz", ".parquet", ".jsonl.gz")):
                errors.append(f"unexpected bulk data or checkout-only file in wheel: {name}")
    with tarfile.open(sources[0]) as source:
        members = source.getnames()
        prefixes = {name.split("/")[0] for name in members}
        if len(prefixes) != 1:
            errors.append("source distribution must have one root directory")
        else:
            prefix = next(iter(prefixes)) + "/"
            expected = {"pyproject.toml", "CITATION.cff", "README.md", "LICENSE",
                        "scripts/check_release.py", "scripts/lint.sh", "scripts/pytest.sh", "Test/test_cli.py"}
            actual = {name.removeprefix(prefix) for name in members}
            errors.extend(f"source distribution missing {name}" for name in sorted(expected - actual))
            stream = source.extractfile(prefix + "pyproject.toml")
            if stream is None or tomllib.loads(stream.read().decode())["project"]["version"] != version:
                errors.append("source distribution version differs from pyproject")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="Require the release tag to equal v<project version>")
    parser.add_argument("--dist-dir", type=Path, help="Inspect a built wheel and source distribution")
    args = parser.parse_args()
    try:
        version, errors = metadata_errors(args.root.resolve(), args.tag)
        if args.dist_dir:
            errors.extend(distribution_errors(args.dist_dir.resolve(), version))
    except (OSError, ValueError, KeyError, SyntaxError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"Release check failed: {error}")
        return 1
    for error in errors:
        print(error)
    if not errors:
        print(f"Release metadata{' and distributions' if args.dist_dir else ''} agree on {version}.")
        print("This check does not certify dataset distribution, reconstruction or completed predictive experiments.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
