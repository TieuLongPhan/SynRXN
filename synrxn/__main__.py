#!/usr/bin/env python3
"""
SynRXN command-line interface.

Typical usage *inside the repository* (no install):

    # Show help
    python -m synrxn --help

    # Build RBL dataset
    python -m synrxn build --rbl

    # Build property dataset
    python -m synrxn build --property

    # Run multiple builders in sequence
    python -m synrxn build --aam --classification --property

    # Forward extra args to builders (after --)
    python -m synrxn build --rbl -- --out-dir Data/rbl

    # CLI-level dry-run: only print commands, do not execute
    python -m synrxn build --rbl --dry-run

    # Ask builders themselves not to save (they must support --dry-run)
    python -m synrxn build --rbl --no-save
"""
from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from synrxn.reconstruction.families._importers import module_for

LOGGER = logging.getLogger(__name__)

DEFAULT_BUILDERS = {
    target: module_for(target)
    for target in ('aam', 'rbl', 'classification', 'synthesis', 'property',
                   'feasibility', 'impurity', 'selectivity', 'yield', 'condition')
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    """Configure a simple stream logger."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    if not LOGGER.handlers:
        LOGGER.addHandler(handler)


def find_repo_root(start: Path) -> Path:
    """
    Locate checkout metadata; installed commands otherwise use the working directory.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file() and (p / "synrxn").is_dir():
            return p
    return Path.cwd()


def build_command(
    python_exe: str,
    script_path: Path,
    forwarded_args: Sequence[str],
) -> List[str]:
    """Build the command list for subprocess."""
    return [python_exe, str(script_path), *forwarded_args]


def run_subprocess(cmd: List[str], cwd: Path, dry_run: bool = False,
                   module_root: Optional[Path] = None) -> int:
    """
    Run subprocess and return exit code.

    If dry_run is True, only print the command that would be run.
    """
    quoted = " ".join(shlex.quote(c) for c in cmd)
    LOGGER.debug("Prepared command: %s", quoted)

    if dry_run:
        print("DRY-RUN (CLI): would execute:", quoted)
        return 0

    try:
        environment = None
        if module_root is not None:
            environment = dict(os.environ)
            previous = environment.get('PYTHONPATH')
            environment['PYTHONPATH'] = str(module_root) + (os.pathsep + previous if previous else '')
        process = subprocess.Popen(cmd, cwd=str(cwd), env=environment)
        process.wait()
        return process.returncode or 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130
    except FileNotFoundError as exc:
        LOGGER.error("Executable not found: %s", exc)
        return 127
    except Exception as exc:  # pragma: no cover (defensive)
        LOGGER.exception("Failed to run command: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="synrxn",
        description="SynRXN dataset access and maintenance commands.",
    )
    from synrxn import __version__
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build subcommand
    build_p = subparsers.add_parser(
        "build", help="Run packaged dataset importers."
    )
    # builder flags (can be combined)
    build_p.add_argument("--aam", action="store_true", help="Run AAM dataset builder.")
    build_p.add_argument("--rbl", action="store_true", help="Run RBL dataset builder.")
    build_p.add_argument("--condition", action="store_true", help="Run condition dataset builder.")
    build_p.add_argument(
        "--feasibility",
        action="store_true",
        help="Run feasibility dataset builder.",
    )
    build_p.add_argument(
        "--impurity",
        action="store_true",
        help="Run impurity dataset builder.",
    )
    build_p.add_argument(
        "--selectivity",
        action="store_true",
        help="Run selectivity dataset builder.",
    )
    build_p.add_argument(
        "--classification",
        action="store_true",
        help="Run classification dataset builder.",
    )
    build_p.add_argument(
        "--synthesis",
        action="store_true",
        help="Run synthesis dataset builder.",
    )
    build_p.add_argument(
        "--property",
        action="store_true",
        help="Run property dataset builder.",
    )
    build_p.add_argument(
        "--yield",
        action="store_true",
        dest="yield_task",
        help="Run yield dataset builder.",
    )

    build_p.add_argument(
        "--script",
        type=str,
        default=None,
        help="Override script path (applies to each target).",
    )
    build_p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run the scripts.",
    )
    build_p.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for builder scripts (defaults to repo root).",
    )
    build_p.add_argument(
        "--dry-run",
        action="store_true",
        help="CLI dry-run: print commands and do not execute them.",
    )
    build_p.add_argument(
        "--no-save",
        action="store_true",
        help="Forward '--dry-run' to builder scripts so they will not save outputs.",
    )
    build_p.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to builder scripts (use -- before them).",
    )

    verify_p = subparsers.add_parser(
        "verify-manifest", help="Verify release files against manifest.json."
    )
    verify_p.add_argument("--manifest", "-m", default="manifest.json")
    verify_p.add_argument("--root", type=str, default=None)
    verify_p.add_argument("--allow-unexpected", action="store_true")
    verify_p.add_argument("--quiet", action="store_true")

    validate_p = subparsers.add_parser(
        "validate", help="Validate dataset metadata, schemas, identifiers, and splits."
    )
    validate_p.add_argument("--data-dir", default="Data")
    validate_p.add_argument("--metadata", default="Data/metadata.yaml")
    validate_p.add_argument("--manifest", default="manifest.json")
    validate_p.add_argument("--quick", action="store_true")
    validate_p.add_argument("--json-output", default=None)

    datasets_p = subparsers.add_parser(
        "datasets", help="List and describe datasets in the packaged catalog."
    )
    datasets_sub = datasets_p.add_subparsers(dest="datasets_command", required=True)
    datasets_list = datasets_sub.add_parser("list", help="List catalog datasets.")
    datasets_list.add_argument("--task", default=None)
    datasets_list.add_argument(
        "--has-split",
        action="store_true",
        help="Show only datasets with published splits.",
    )
    datasets_list.add_argument("--json", action="store_true", dest="as_json")
    datasets_describe = datasets_sub.add_parser(
        "describe", help="Describe one dataset."
    )
    datasets_describe.add_argument("task")
    datasets_describe.add_argument("name")
    datasets_describe.add_argument("--json", action="store_true", dest="as_json")

    parquet_p = subparsers.add_parser(
        "parquet", help="Build or verify deterministic Parquet query artifacts."
    )
    parquet_sub = parquet_p.add_subparsers(dest="parquet_command", required=True)
    parquet_build = parquet_sub.add_parser("build", help="Build all Parquet artifacts.")
    parquet_build.add_argument("--data-dir", default="Data")
    parquet_build.add_argument("--output-dir", required=True)
    parquet_build.add_argument("--manifest", default="manifest.json")
    parquet_build.add_argument("--partition", choices=("main", "sharealike"))
    parquet_verify = parquet_sub.add_parser(
        "verify", help="Verify Parquet artifacts against canonical CSV files."
    )
    parquet_verify.add_argument("--data-dir", default="Data")
    parquet_verify.add_argument("--parquet-dir", required=True)

    catalog_p = subparsers.add_parser(
        "catalog-assets", help="Build bounded static catalog UI assets."
    )
    catalog_p.add_argument("--data-dir", default="Data")
    catalog_p.add_argument("--metadata", default="Data/metadata.yaml")
    catalog_p.add_argument("--manifest", default="manifest.json")
    catalog_p.add_argument("--output", default="doc/_static/catalog-data.json")
    catalog_p.add_argument("--reaction-dir", default="doc/_static/catalog-reactions")

    references_p = subparsers.add_parser(
        "references", help="Inspect a checksum-pinned observation/reference release.")
    references_p.add_argument("--store", type=Path, required=True)
    references_p.add_argument("--accession")
    references_p.add_argument("--evidence-tier")
    references_p.add_argument("--offset", type=int, default=0)
    references_p.add_argument("--limit", type=int, default=20)

    evaluate_p = subparsers.add_parser(
        "evaluate", help="Score archived predictions under an explicit task contract.")
    evaluate_p.add_argument("--contract", type=Path, required=True)
    evaluate_p.add_argument("--instances", type=Path, required=True)
    evaluate_p.add_argument("--references", type=Path, required=True)
    evaluate_p.add_argument("--predictions", type=Path, required=True)
    evaluate_p.add_argument("--membership", type=Path,
                            help="Select the exact records from a versioned split manifest")
    evaluate_p.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    evaluate_p.add_argument("--output", type=Path)

    task_p = subparsers.add_parser("task", help="Inspect or export one of the eight input/target task views.")
    task_p.add_argument("action", choices=("list", "show", "export"))
    task_p.add_argument("--task-id")
    task_p.add_argument("--dataset", help="Canonical key such as feasibility/frea_expert.csv.gz")
    task_p.add_argument("--data-dir", type=Path, default=Path("Data"))
    task_p.add_argument("--output", type=Path)
    task_p.add_argument("--seed", type=int, default=42)
    task_p.add_argument("--no-proposed-split", action="store_true",
                        help="Export for review before choosing an evaluation split")
    task_p.add_argument("--limit", type=int, help="Bounded preview; omitted for a complete data view")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "task":
        from synrxn.benchmarks.task_views import definitions, export_task_view, task_definition

        try:
            if args.action == "list":
                result = [{"id": task["id"], "name": task["name"]} for task in definitions()["tasks"]]
            elif args.action == "show":
                if not args.task_id:
                    raise ValueError("--task-id is required")
                result = task_definition(args.task_id)
            else:
                if not args.task_id or not args.dataset or not args.output:
                    raise ValueError("export requires --task-id, --dataset and --output")
                result = export_task_view(args.task_id, args.dataset, args.data_dir, args.output,
                                          seed=args.seed, limit=args.limit, propose_split=not args.no_proposed_split)
            print(json.dumps(result, indent=2))
            return 0
        except (ValueError, KeyError, OSError) as exc:
            LOGGER.error("Task-view command failed: %s", exc)
            return 2

    if args.command == "references":
        from synrxn.benchmarks.reference_store import ReferenceStore

        try:
            store = ReferenceStore(args.store)
            result = store.inspect(args.accession) if args.accession else store.list(
                evidence_tier=args.evidence_tier, offset=args.offset, limit=args.limit)
        except (ValueError, KeyError, OSError) as exc:
            LOGGER.error("Reference inspection failed: %s", exc)
            return 2
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "evaluate":
        from synrxn.metrics.reference import evaluate_predictions
        from synrxn.benchmarks.contracts import ReferenceRecord, TaskContract
        from synrxn.benchmarks.reference_store import read_jsonl
        import hashlib

        try:
            contract = TaskContract.from_dict(json.loads(args.contract.read_text()))
            instances = read_jsonl(args.instances)
            if args.membership:
                membership = json.loads(args.membership.read_text())["membership"]
                unknown = set(membership) - {row["instance_id"] for row in instances}
                if unknown:
                    raise ValueError("split manifest contains unknown task instances")
                instances = [row for row in instances
                             if membership.get(row["instance_id"]) == args.partition]
            result = evaluate_predictions(
                instances,
                [ReferenceRecord.from_dict(r) for r in read_jsonl(args.references)],
                read_jsonl(args.predictions), contract)
            result["input_sha256"] = {
                name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                for name in ("contract", "instances", "references", "predictions")}
            if args.membership:
                result["input_sha256"]["membership"] = hashlib.sha256(args.membership.read_bytes()).hexdigest()
                result["partition"] = args.partition
            rendered = json.dumps(result, indent=2) + "\n"
            if args.output:
                if args.output.exists():
                    raise ValueError("score output exists; choose a new path to preserve prior scoring")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered)
            else:
                print(rendered, end="")
        except (ValueError, KeyError, OSError) as exc:
            LOGGER.error("Evaluation failed: %s", exc)
            return 2
        return 0

    if args.command == "verify-manifest":
        from synrxn.data.verify_manifest import main as verify_manifest_main

        verify_argv = ["--manifest", args.manifest]
        if args.root:
            verify_argv.extend(["--root", args.root])
        if args.allow_unexpected:
            verify_argv.append("--allow-unexpected")
        if args.quiet:
            verify_argv.append("--quiet")
        return verify_manifest_main(verify_argv)

    if args.command == "validate":
        from synrxn.data.validate import main as validate_data_main

        validate_argv = [
            "--data-dir",
            args.data_dir,
            "--metadata",
            args.metadata,
            "--manifest",
            args.manifest,
        ]
        if args.quick:
            validate_argv.append("--quick")
        if args.json_output:
            validate_argv.extend(["--json-output", args.json_output])
        return validate_data_main(validate_argv)

    if args.command == "datasets":
        from .data import DatasetCatalog

        catalog = DatasetCatalog()
        if args.datasets_command == "list":
            records = catalog.list(
                task=args.task, has_split=True if args.has_split else None
            )
            if args.as_json:
                print(json.dumps([asdict(record) for record in records], indent=2))
            else:
                for record in records:
                    splits = ",".join(record.split_values) or "—"
                    print(
                        f"{record.task.value}/{record.name}\t{record.title}\t"
                        f"targets={','.join(record.targets) or '—'}\tsplits={splits}"
                    )
            return 0
        try:
            record = catalog.get(args.task, args.name)
        except (KeyError, ValueError) as exc:
            LOGGER.error("%s", exc.args[0] if exc.args else str(exc))
            return 2
        if args.as_json:
            print(json.dumps(asdict(record), indent=2))
        else:
            print(f"{record.task.value}/{record.name}: {record.title}")
            print(record.description)
            print(f"Role: {record.benchmark_role}")
            print(f"Targets: {', '.join(record.targets) or 'none declared'}")
            print(f"Splits: {', '.join(record.split_values) or 'none published'}")
            print(f"License: {record.license}")
            print(f"Citations: {', '.join(record.citations)}")
        return 0

    if args.command == "parquet":
        from synrxn.access.query import main as query_main

        query_argv = [
            args.parquet_command,
            "--data-dir",
            args.data_dir,
        ]
        if args.parquet_command == "build":
            query_argv.extend(
                ["--output-dir", args.output_dir, "--manifest", args.manifest]
            )
            if args.partition:
                query_argv.extend(["--partition", args.partition])
        else:
            query_argv.extend(["--parquet-dir", args.parquet_dir])
        return query_main(query_argv)

    if args.command == "catalog-assets":
        from synrxn.data.build_catalog import main as catalog_main

        return catalog_main(
            [
                "--data-dir",
                args.data_dir,
                "--metadata",
                args.metadata,
                "--manifest",
                args.manifest,
                "--output",
                args.output,
                "--reaction-dir",
                args.reaction_dir,
            ]
        )

    main_path = Path(__file__).resolve()
    # main.py lives in synrxn/, repo root is one level up (or higher if we autodetect)
    repo_root = find_repo_root(main_path.parent)

    # Collect requested targets (in order)
    targets: List[str] = []
    if args.aam:
        targets.append("aam")
    if args.rbl:
        targets.append("rbl")
    if args.classification:
        targets.append("classification")
    if args.synthesis:
        targets.append("synthesis")
    if args.property:
        targets.append("property")
    if args.condition:
        targets.append("condition")
    if args.feasibility:
        targets.append("feasibility")
    if args.impurity:
        targets.append("impurity")
    if args.selectivity:
        targets.append("selectivity")
    if args.yield_task:
        targets.append("yield")

    if not targets:
        LOGGER.error(
            "No build target provided. Use one or more of "
            "--aam, --rbl, --classification, --synthesis, --property, --condition, "
            "--feasibility, --impurity, --selectivity, or --yield."
        )
        return 2

    # Forwarded args: remove leading "--" that argparse keeps for REMAINDER
    forwarded_base: List[str] = (
        [a for a in args.forwarded if a != "--"] if args.forwarded else []
    )

    # If we want builders to avoid saving, append --dry-run for them
    forwarded_for_builder = list(forwarded_base)
    if args.no_save and "--dry-run" not in forwarded_for_builder:
        forwarded_for_builder.append("--dry-run")

    # Resolve python executable
    python_exe = shutil.which(args.python) or args.python
    if not shutil.which(python_exe) and not Path(python_exe).exists():
        LOGGER.warning(
            "Python executable %s not found in PATH; attempting to use it directly.",
            python_exe,
        )

    # Working directory for subprocesses
    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else repo_root
    if not cwd.exists():
        LOGGER.error("Working directory does not exist: %s", cwd)
        return 3

    overall_rc = 0

    for target in targets:
        if args.script is not None:
            script_path = Path(args.script).expanduser().resolve()
            if not script_path.exists():
                LOGGER.error("Target script for '%s' not found: %s", target, script_path)
                overall_rc = overall_rc or 4
                continue
            cmd = build_command(python_exe, script_path, forwarded_for_builder)
        else:
            cmd = [python_exe, '-m', DEFAULT_BUILDERS[target], *forwarded_for_builder]

        LOGGER.info(
            "Running target '%s': %s",
            target,
            " ".join(shlex.quote(c) for c in cmd),
        )
        LOGGER.debug("Working directory: %s", cwd)

        rc = run_subprocess(cmd, cwd=cwd, dry_run=args.dry_run, module_root=repo_root)  # type: ignore
        if rc != 0:
            LOGGER.error("Target '%s' exited with code %d", target, rc)
            overall_rc = rc

    if overall_rc == 0:
        LOGGER.info("All requested builders finished successfully.")
    else:
        LOGGER.error(
            "One or more builders failed. Last non-zero exit code: %d", overall_rc
        )

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
