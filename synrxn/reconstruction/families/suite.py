#!/usr/bin/env python3
"""Reproduce and audit the available twelve-family SynRXN research artifacts.

Offline by default and never a model-training runner. Each run gets a fresh
directory, frozen code/input snapshots, file checks, task readiness and logs.
Python >=3.11; validation tests require pytest. --check-rows additionally needs
pyarrow for Parquet. See doc/data_concept.rst for acquisition and benchmark boundaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from synrxn.reconstruction.paths import REPO, source_path, physical_path as resolve_path, logical_relative
KDD = resolve_path("kdd")
REGISTRY_OUTPUTS = ["task_registry.json"]
HITEA_OUTPUTS = ["memberships.jsonl.gz", "exclusions.json", "panel_cards.csv", "audit.json"]


def physical_sha(path: Path) -> str:
    """Hash actual file bytes without applying historical source aliases."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha(path: Path) -> str:
    return physical_sha(source_path(path))


def read_json(path: Path):
    return json.loads(resolve_path(path).read_text(encoding="utf8"))


def write_json(path: Path, value):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def local_path(repo: Path, relative: str) -> Path:
    path = source_path(relative, repo=repo).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(repo.resolve()):
        raise ValueError("Source path resolves outside the repository: " + relative)
    return path


def dataset_inventory(repo: Path) -> list[dict]:
    records = []
    for name in ["catalog-data.json", "catalog-sharealike-data.json"]:
        for item in read_json(repo / "doc/_static" / name)["datasets"]:
            records.append({"namespace": "canonical", "key": item["id"],
                            "path": "Data/" + item["id"] + ".csv.gz", "format": "csv.gz",
                            "expected_sha256": item["sha256"], "expected_bytes": item["size"],
                            "expected_rows": item["rows"], "role": "primary"})
    for item in read_json(repo / "Data/expansion/catalog.json")["tables"]:
        records.append({"namespace": "expansion", "key": item["key"],
                        "path": (Path("Data/expansion") / item["path"]).as_posix(), "format": "parquet",
                        "expected_sha256": item["sha256"], "expected_bytes": item["bytes"],
                        "expected_rows": item["rows"], "role": item["role"]})
    for catalog_path in sorted(resolve_path("Data/coverage_expansion", repo).glob("*/catalog.json")):
        for item in read_json(catalog_path)["tables"]:
            path = item["path"]
            fmt = "parquet" if path.endswith(".parquet") else "jsonl.gz" if path.endswith(".jsonl.gz") else "csv"
            records.append({"namespace": "coverage", "key": catalog_path.parent.name + "/" + item["id"],
                            "path": path, "format": fmt, "expected_sha256": item["sha256"],
                            "expected_bytes": item.get("bytes"), "expected_rows": item["rows"],
                            "role": item.get("role", "primary")})
    if len(records) != len({(item["namespace"], item["key"]) for item in records}):
        raise ValueError("Duplicate dataset identity in catalogs")
    return records


def audit_file(record: dict, repo: Path, check_rows: bool = False) -> dict:
    result = {**record, "status": "failed", "errors": [], "observed_sha256": None,
              "observed_bytes": None, "observed_rows": None,
              "row_count_status": "catalog_declared" if "expected_rows" in record else "not_requested"}
    try:
        path = local_path(repo, record["path"])
        if path != (repo / record["path"]).resolve():
            result["resolved_source_path"] = path.relative_to(repo).as_posix()
            result["source_path_kind"] = "byte-frozen historical code alias"
        before = path.stat()
        result["observed_bytes"] = before.st_size
        result["observed_sha256"] = sha(path)
        if result["observed_sha256"] != record["expected_sha256"]:
            result["errors"].append("SHA-256 mismatch")
        if record.get("expected_bytes") is not None and before.st_size != record["expected_bytes"]:
            result["errors"].append("Byte-size mismatch")
        if check_rows and "expected_rows" in record:
            result["row_count_status"] = "failed"
            if record.get("format") in {"csv.gz", "csv"}:
                csv.field_size_limit(16 * 1024 * 1024)
                opener = gzip.open if record["format"] == "csv.gz" else open
                with opener(path, "rt", encoding="utf8", newline="") as handle:
                    reader = csv.reader(handle)
                    if next(reader, None) is None:
                        raise ValueError("CSV has no header")
                    result["observed_rows"] = sum(bool(row) for row in reader)
            elif record.get("format") == "jsonl.gz":
                with gzip.open(path, "rt", encoding="utf8") as handle:
                    result["observed_rows"] = sum(isinstance(json.loads(line), dict) for line in handle)
            elif record.get("format") == "parquet":
                try:
                    import pyarrow.parquet as pq
                except ImportError as error:
                    raise ValueError("--check-rows needs pyarrow for Parquet") from error
                result["observed_rows"] = pq.ParquetFile(path).metadata.num_rows
            else:
                raise ValueError("Unsupported row-count format: " + str(record.get("format")))
            if result["observed_rows"] != record["expected_rows"]:
                result["errors"].append("Row-count mismatch")
            else:
                result["row_count_status"] = "verified"
        after = path.stat()
        result["observed_mtime_ns"] = after.st_mtime_ns
        result["observed_inode"] = after.st_ino
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            result["errors"].append("Source changed during audit")
    except Exception as error:
        result["errors"].append(str(error))
    result["status"] = "failed" if result["errors"] else "verified"
    return result


def legacy_inventory(repo: Path) -> list[dict]:
    base = Path("ncs/artifacts/task-views-002")
    records = []
    for export in read_json(repo / base / "summary.json")["exports"]:
        report = export["report"]
        if not report.get("files_sha256"):
            raise ValueError("Legacy export has no file hashes: " + export["path"])
        for filename, digest in report["files_sha256"].items():
            records.append({"namespace": "legacy_export", "key": export["path"] + "/" + filename,
                            "path": (base / export["path"] / filename).as_posix(),
                            "expected_sha256": digest, "legacy_view_id": report["task"],
                            "dataset_key": report["dataset"].removesuffix(".csv.gz")})
    return records


def build_task_matrix(registry: dict, audits: list[dict]) -> list[dict]:
    verified = {(item["namespace"], item["key"]): item for item in audits}
    rows = []
    for state, tasks in [("source_backed", registry["tasks"]), ("roadmap", registry["future_tasks"])]:
        for task in tasks:
            bindings = {(item["namespace"], item["key"]): item for item in task.get("bindings", [])}
            primary = [item for item in bindings.values() if item["role"] != "overlapping_view"]
            missing = [f"{namespace}:{key}" for namespace, key in bindings
                       if verified.get((namespace, key), {}).get("status") != "verified"]
            blockers = ["Unverified source: " + key for key in missing]
            if state == "roadmap":
                blockers += ["Acquire and qualify task references", task.get("required_data", "Task references required")]
            blockers += ["Freeze the task-specific qualified contract and baseline configuration",
                         "Run and archive predictive evaluation; this workflow fits no models"]
            rows.append({"id": task["id"], "title": task["title"], "family_id": task["family_id"],
                         "data_status": state,
                         "source_verification": "not_available" if state == "roadmap" else "failed" if missing or not bindings else "verified",
                         "primary_table_count": len(primary), "primary_source_rows": sum(item["rows"] for item in primary),
                         "legacy_view_ids": task.get("legacy_view_ids", []), "interface_status": task.get("readiness"),
                         "benchmark_status": task["benchmark_result_status"], "blockers": blockers,
                         "release_eligible_binding_count": sum(bool(item.get("release_eligible", True)) for item in bindings.values()),
                         "release_readiness": task.get("release_readiness", "see source-rights audit"),
                         "question": task.get("question"), "metrics_or_evaluation": task.get("metrics", task.get("evaluation")),
                         "qualification": task.get("qualification", task.get("required_data"))})
    return rows


def compare_artifacts(expected: Path, actual: Path, names: list[str]) -> list[dict]:
    results = []
    for name in names:
        record = {"name": name, "status": "failed", "errors": [], "expected_sha256": None, "observed_sha256": None}
        try:
            record["expected_sha256"] = sha(expected / name)
            record["observed_sha256"] = sha(actual / name)
            if record["expected_sha256"] != record["observed_sha256"]:
                record["errors"].append("Regenerated artifact differs from the frozen artifact")
        except OSError as error:
            record["errors"].append(str(error))
        record["status"] = "failed" if record["errors"] else "verified"
        results.append(record)
    return results


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key in fields})


def metadata_pins(registry: dict, repo: Path, preparation: bool) -> list[dict]:
    pins = dict(registry["source_sha256"])
    expansion = read_json(repo / "Data/expansion/catalog.json")
    for field, relative in [("survey_sha256", "kdd/integration/input_survey.md"),
                            ("source_audit_sha256", "kdd/integration/source_audit.json"),
                            ("legacy_manifest_sha256", "manifest.json")]:
        pins[relative] = expansion[field]
    if preparation:
        case = read_json(repo / "kdd/case_studies/hitea_protocol.json")
        pins["kdd/preparation/raw/hitea.csv"] = case["population"]["upstream_sha256"]
    for path in sorted(resolve_path("kdd/coverage", repo).rglob("*.json")):
        if path.name not in {"release_audit.json", "release_manifest.json"}:
            pins[logical_relative(path, repo)] = sha(path)
    for path in sorted(resolve_path("Data/coverage_expansion", repo).glob("*/catalog.json")):
        pins[logical_relative(path, repo)] = sha(path)
    return [{"namespace": "pinned_metadata", "key": name, "path": name, "expected_sha256": digest}
            for name, digest in sorted(pins.items())]


def snapshot(run: Path, records: list[dict], repo: Path, preparation: bool = False) -> list[dict]:
    names = {item["path"] for item in records if item["namespace"] == "pinned_metadata" and not item["path"].endswith(".csv")}
    names |= {path.relative_to(repo).as_posix() for path in (repo / "Test/reconstruction").rglob("*.py")}
    names |= {logical_relative(path, repo) for path in resolve_path('kdd/coverage', repo).rglob('*')
              if path.is_file() and path.suffix in {'.md', '.json', '.txt'} and '__pycache__' not in path.parts}
    # Runtime imports span source parsers, access, task contracts and metrics.
    # Capture their actual maintained files separately from historical pins.
    names |= {path.relative_to(repo).as_posix() for path in (repo / "synrxn").rglob("*.py")}
    names |= {"pyproject.toml", "kdd/study_protocol.json", "kdd/task_registry.json",
              "kdd/scope_policy.json", "doc/data_concept.rst", "doc/task_evaluation.rst"}
    names |= {logical_relative(path, repo) for pattern in ["*.md", "*.json", "*.csv", "requirements*.txt"]
              for path in resolve_path("kdd", repo).glob(pattern)}
    names |= {"doc/reference.rst", "ncs/baselines/v1.2.1/README.md"}
    if preparation:
        names |= {"kdd/literature/" + name for name in ["selected_papers.json", "downloads.json", "full_text_xml.json",
                  "reviews.json", "track_screening.csv", "track_inventory.json", "references.bib"]}
        names |= {"kdd/preparation/" + name for name in HITEA_OUTPUTS}
    names |= {"script/" + name for name in ["build_expansion_dataset.py", "fetch_expansion_data.py", "audit_expansion_sources.py"]}
    if resolve_path("kdd/case_studies", repo).exists():
        names |= {logical_relative(path, repo) for path in resolve_path("kdd/case_studies", repo).glob("*.json")}
    return snapshot_physical_files(run, names, repo)


def physical_path(repo: Path, relative: str) -> Path:
    """Resolve an actual repository file, retaining compatibility-shim bytes."""
    root = repo.resolve()
    path = resolve_path(relative, repo=root).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Physical source is missing or escapes the repository: " + relative)
    return path


def snapshot_physical_files(run: Path, names, repo: Path) -> list[dict]:
    """Snapshot actual files and their explicit archived source counterparts.

    Historical pins still resolve through local_path(). A physical entry point
    is a separate runtime input and must never be replaced by its old kernel.
    """
    root = repo.resolve()
    expanded = set(names)
    aliases = {}
    retired = {}
    for name in sorted(expanded):
        historical = local_path(root, name).resolve()
        if not resolve_path(name, root).is_file():
            # A removed launcher is a logical provenance key, not a missing
            # runtime file. Snapshot its explicitly preserved original and
            # retain the resolved physical path for independent post-checks.
            if historical == root / name:
                raise ValueError('Missing source has no preserved original: ' + name)
            physical_path(root, historical.relative_to(root).as_posix())
            retired[name] = historical.relative_to(root).as_posix()
            continue
        physical = physical_path(root, name)
        if historical != physical:
            historical_name = historical.relative_to(root).as_posix()
            aliases[name] = historical_name
            expanded.add(historical_name)
    result = []
    for name in sorted(expanded):
        source = physical_path(root, retired.get(name, name))
        destination = run / "snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = physical_sha(destination)
        if physical_sha(source) != digest:
            raise ValueError("Physical source changed while taking snapshot: " + name)
        record = {"path": name, "resolved_source_path": source.relative_to(root).as_posix(),
                  "sha256": digest, "bytes": destination.stat().st_size,
                  "hash_semantics": "physical_file_bytes_without_source_aliases"}
        if name in retired:
            record['logical_path_status'] = 'retired; resolved to preserved original'
        if name in aliases:
            historical_name = aliases[name]
            record["historical_source_alias"] = {
                "logical_source_path": name, "resolved_source_path": historical_name,
                "sha256": physical_sha(physical_path(root, historical_name))}
        result.append(record)
    return result


def physical_snapshot_changes(records: list[dict], repo: Path) -> list[str]:
    """Detect changed runtime files independently of their frozen code pins."""
    changed = []
    for record in records:
        try:
            retired = record.get('logical_path_status') == 'retired; resolved to preserved original'
            path = physical_path(repo, record['resolved_source_path'] if retired else record["path"])
            if (path.relative_to(repo.resolve()).as_posix() != record["resolved_source_path"]
                    or physical_sha(path) != record["sha256"]
                    or (retired and (repo / record['path']).exists())):
                changed.append(record["path"])
        except (OSError, ValueError):
            changed.append(record["path"])
    return sorted(set(changed))


def environment(repo: Path) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = sorted({(item.metadata["Name"], item.version) for item in importlib.metadata.distributions() if item.metadata["Name"]})
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "packages": [{"name": name, "version": version} for name, version in packages],
            "subprocess_environment": {"PYTHONHASHSEED": "2027", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}}


def execute(step_id: str, argv: list[str], run: Path, manifest: dict, env: dict, timeout: float):
    print("Running " + step_id, flush=True)
    log = run / "logs" / (step_id + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": step_id, "argv": argv, "cwd": str(REPO), "log": log.relative_to(run).as_posix(), "status": "running"}
    manifest["steps"].append(record)
    start = time.monotonic()
    try:
        with log.open("w", encoding="utf8") as handle:
            completed = subprocess.run(argv, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        record["returncode"] = completed.returncode
        record["status"] = "passed" if completed.returncode == 0 else "failed"
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(status="failed", returncode=None, error=str(error))
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    write_json(run / "manifest.json", manifest)
    if record["status"] != "passed":
        raise ValueError("Step failed: " + step_id + "; inspect " + str(log))


def task_dataset_plan(registry: dict, suite: dict) -> list[dict]:
    rows = []
    for task in registry["tasks"]:
        for item in task["bindings"]:
            rows.append({"task_id": task["id"], "family_id": task["family_id"], **item,
                         "source_rows": item["rows"], "metrics": item.get('task_fields', {}).get(task['id'], {}).get('metrics', task["metrics"]),
                         "task_metrics": task["metrics"],
                         "qualification": item.get("qualification", task["qualification"]),
                         "task_qualification": task["qualification"],
                         "baseline_policy": suite["family_baselines"][task["family_id"]],
                         "seeds": suite["compute_allocation"]["stochastic_final_seeds"],
                         "execution_status": "pending qualification and predictive runs" if item.get('release_eligible', True)
                         else "excluded from release execution; source rights unresolved or blocked"})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["scope", "preparation"], default="preparation",
                        help="scope: data/exports/taxonomy/tests; preparation: also regenerate HiTEA and validate research artifacts")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory outside source/data trees; default: .cache/synrxn/runs/full/<UTC timestamp>")
    parser.add_argument("--check-rows", action="store_true", help="Also stream CSV rows and read Parquet row metadata; slower and requires pyarrow")
    parser.add_argument("--rebuild-coverage", action="store_true", help="Also rebuild every acquired coverage adapter in an isolated offline workspace and compare exact bytes")
    parser.add_argument("--step-timeout", type=float, default=1800, help="Per-subprocess timeout in seconds (default: 1800)")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python >=3.11 is required")
    if args.step_timeout <= 0:
        parser.error("--step-timeout must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = (args.output_dir or REPO / ".cache/synrxn/runs/full" / stamp).resolve()
    protected = [REPO / name for name in ['script', 'kdd', 'ncs', 'synrxn', 'Data', 'archive/research', 'paper/kdd/literature']]
    if run == REPO or any(run.is_relative_to(source) for source in protected) or (run.exists() and (not run.is_dir() or any(run.iterdir()))):
        parser.error("Use a new or empty output directory outside source/data trees; existing artifacts are never overwritten")
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "synrxn.kdd-reproduction.v1", "started_at": stamp, "profile": args.profile,
                "status": "running", "network_access": "none requested by this workflow", "steps": [], "errors": [],
                "predictive_evaluation": "pending; no models are fitted or scored by this workflow",
                "completed_benchmarks_in_this_run": 0, "row_recount_requested": args.check_rows,
                "historical_scope": {"published_families": 5, "reference": "doc/reference.rst",
                                     "meaning": "The original five were broad families; the current eight exports are interfaces"},
                "invocation": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]}
    write_json(run / "manifest.json", manifest)
    registry, audits, snapshots, pin_checks = None, [], [], []
    all_audited, comparisons = [], {}
    start = time.monotonic()
    try:
        registry = read_json(KDD / "task_registry.json")
        manifest["counts"] = {**registry["summary"], "total_task_definitions": len(registry["tasks"]) + len(registry["future_tasks"])}
        manifest["excluded_tasks"] = registry["excluded_tasks"]
        pins = metadata_pins(registry, REPO, args.profile == "preparation")
        pin_checks = [audit_file(record, REPO) for record in pins]
        write_json(run / "source_pins.json", pin_checks)
        if any(item["status"] != "verified" for item in pin_checks):
            raise ValueError("Pinned input changed or is missing; inspect source_pins.json before rebuilding or repinning")
        snapshots = snapshot(run, pins, REPO, preparation=args.profile == "preparation")
        write_json(run / "snapshot_manifest.json", snapshots)
        env_record = environment(REPO)
        write_json(run / "environment.json", env_record)
        env = {**os.environ, **env_record["subprocess_environment"]}
        for label, records, recount in [("datasets", dataset_inventory(REPO), args.check_rows),
                                         ("legacy_exports", legacy_inventory(REPO), False)]:
            print(f"Verifying {len(records)} {label} files", flush=True)
            checks = []
            for record in records:
                checks.append(audit_file(record, REPO, recount))
            write_json(run / (label + "_verification.json"), checks)
            all_audited += checks
            if label == "datasets":
                audits = checks
            if any(item["status"] != "verified" for item in checks):
                raise ValueError("Source verification failed; inspect " + label + "_verification.json")
        manifest["verified_dataset_files"] = len(audits)
        manifest["verified_legacy_export_files"] = sum(item["namespace"] == "legacy_export" for item in all_audited)
        out = run / "artifacts/registry"
        execute("build_registry", [sys.executable, "-m", "synrxn.reconstruction", "registry", "--output", str(out)], run, manifest, env, args.step_timeout)
        comparisons["registry"] = compare_artifacts(KDD, out, REGISTRY_OUTPUTS)
        write_json(run / "artifact_comparisons.json", comparisons)
        if any(item["status"] != "verified" for item in comparisons["registry"]):
            raise ValueError("Registry regeneration differs from the frozen snapshot; inspect artifact_comparisons.json")
        execute("coverage_release_audit", [sys.executable, "-m", "synrxn.reconstruction.coverage.verify_release",
                 "--output", str(run / "artifacts/coverage/release_audit.json"),
                 *([] if args.check_rows else ["--no-rows"])], run, manifest, env, args.step_timeout)
        if args.check_rows:
            execute('content_quality', [sys.executable, '-m', 'synrxn.reconstruction.coverage.quality',
                    '--output', str(run / 'artifacts/coverage/content_quality.json')], run, manifest, env, args.step_timeout)
        if args.rebuild_coverage:
            execute("rebuild_coverage", [sys.executable, "-m", "synrxn.reconstruction.coverage.rebuild",
                     "--output", str(run / "coverage_rebuild")], run, manifest, env, args.step_timeout)
        test_files = [
            "Test/reconstruction/catalogue/test_registry.py",
            "Test/reconstruction/families/test_suite.py",
            "Test/reconstruction/families",
            "Test/reconstruction/provenance",
            "Test/reconstruction/coverage",
            "Test/reconstruction/expansion",
        ]
        if args.profile == "preparation":
            hitea = run / "artifacts/hitea"
            execute("prepare_hitea", [sys.executable, "-m", "synrxn.reconstruction", "prepare-hitea", "--output", str(hitea)], run, manifest, env, args.step_timeout)
            comparisons["hitea"] = compare_artifacts(KDD / "preparation", hitea, HITEA_OUTPUTS)
            write_json(run / "artifact_comparisons.json", comparisons)
            if any(item["status"] != "verified" for item in comparisons["hitea"]):
                raise ValueError("HiTEA regeneration differs from the captured preparation; original review files were not modified")
        execute("tests", [sys.executable, "-m", "pytest", "-q", *test_files], run, manifest, env, args.step_timeout)
        if args.profile == "preparation":
            execute("validate_plan", [sys.executable, "-m", "synrxn.reconstruction", "validate-plan",
                                      "--output", str(run / "artifacts/protocol_validation.json")],
                    run, manifest, env, args.step_timeout)
            execute("validate_literature", [sys.executable, "-m", "synrxn.reconstruction", "validate-literature",
                                          "--output", str(run / "artifacts/literature_validation.json")],
                    run, manifest, env, args.step_timeout)
        manifest["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        changed = []
        try:
            # Runtime snapshots hash physical shim and kernel bytes separately.
            manifest["physical_post_run_changes"] = physical_snapshot_changes(snapshots, REPO)
            changed.extend(manifest["physical_post_run_changes"])
            # Historical source pins retain their explicit archived resolution.
            for item in [{"path": pin["path"], "sha256": pin["expected_sha256"]} for pin in pin_checks]:
                try:
                    if sha(local_path(REPO, item["path"])) != item["sha256"]:
                        changed.append(item["path"])
                except OSError:
                    changed.append(item["path"])
            # Data were hashed above; detect subsequent edits/replacements via stat.
            for item in all_audited:
                try:
                    stat = local_path(REPO, item["path"]).stat()
                    if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (item["observed_bytes"], item.get("observed_mtime_ns"), item.get("observed_inode")):
                        changed.append(item["path"])
                except (OSError, ValueError):
                    changed.append(item["path"])
        except Exception as error:
            manifest["status"] = "failed"
            manifest["errors"].append("Source finalization failed: " + str(error))
        manifest["post_run_check_policy"] = "Runtime snapshots: physical SHA-256 without aliases; historical metadata/code/raw HiTEA pins: alias-aware SHA-256; previously hashed dataset/export files: size, mtime and inode"
        manifest["post_run_changed_sources"] = sorted(set(changed))
        if changed:
            manifest["status"] = "failed"
            manifest["errors"].append("Inputs or code changed during reproduction")
        try:
            if registry is not None:
                matrix = build_task_matrix(registry, audits)
                write_json(run / "task_readiness.json", matrix)
                write_csv(run / "task_readiness.csv", matrix,
                          ["family_id", "id", "title", "data_status", "source_verification", "primary_table_count",
                           "primary_source_rows", "legacy_view_ids", "interface_status", "benchmark_status", "blockers"])
                manifest["task_readiness_counts"] = dict(Counter(row["data_status"] for row in matrix))
                suite = read_json(KDD / "study_protocol.json")
                plan = task_dataset_plan(registry, suite)
                write_json(run / "task_dataset_plan.json", plan)
                write_csv(run / "task_dataset_plan.csv", plan,
                          ["family_id", "task_id", "namespace", "key", "path", "role", "source_rows", "counting_unit", "legacy_field_eligible_rows",
                           "release_eligible", "release_license_expression", "partition", "commercial_use_allowed", "rights_status", "adapter_ids", "metrics", "task_fields", "qualification", "task_qualification", "baseline_policy", "seeds", "execution_status"])
        except Exception as error:
            manifest["status"] = "failed"
            manifest["errors"].append("Task-report finalization failed: " + str(error))
        manifest["elapsed_seconds"] = round(time.monotonic() - start, 3)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        artifacts = []
        try:
            for path in sorted(run.rglob("*")):
                if path.is_file() and path != run / "manifest.json":
                    artifacts.append({"path": path.relative_to(run).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
        except Exception as error:
            manifest["status"] = "failed"
            manifest["errors"].append("Artifact-hash finalization failed: " + str(error))
        manifest["output_artifacts"] = artifacts
        write_json(run / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_dir": str(run), "counts": manifest.get("counts"),
                      "completed_benchmarks_in_this_run": 0, "errors": manifest["errors"]}), flush=True)
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
