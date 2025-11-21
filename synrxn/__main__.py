#!/usr/bin/env python3
"""
SynRXN CLI (synrxn/main.py)

Typical usage *inside the repository* (no install):

    # Show help
    python -m synrxn.main --help

    # Build RBL dataset
    python -m synrxn.main build --rbl

    # Build property dataset
    python -m synrxn.main build --property

    # Run multiple builders in sequence
    python -m synrxn.main build --aam --classification --property

    # Forward extra args to builders (after --)
    python -m synrxn.main build --rbl -- --out-dir Data/rbl

    # CLI-level dry-run: only print commands, do not execute
    python -m synrxn.main build --rbl --dry-run

    # Ask builders themselves not to save (they must support --dry-run)
    python -m synrxn.main build --rbl --no-save
"""
from __future__ import annotations

import argparse
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

# Map logical targets to their default script locations
DEFAULT_SCRIPTS = {
    "aam": "script/build_aam_dataset.py",
    "rbl": "script/build_rbl_dataset.py",
    "classification": "script/build_classification_dataset.py",
    "synthesis": "script/build_synthesis_dataset.py",
    "property": "script/build_property_dataset.py",
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
    Try to detect the repository root starting from the given path.

    Strategy:
    - Walk upwards until we find a directory that contains a "script" subdir.
    - Fallback to "start" if nothing is found.
    """
    for p in [start, *start.parents]:
        if (p / "script").is_dir():
            return p
    return start


def find_script_path(repo_root: Path, override: Optional[str], target: str) -> Path:
    """Resolve the script path for a given target (possibly overridden)."""
    if override is not None:
        return Path(override).expanduser().resolve()

    rel = DEFAULT_SCRIPTS[target]
    return (repo_root / rel).resolve()


def build_command(
    python_exe: str,
    script_path: Path,
    forwarded_args: Sequence[str],
) -> List[str]:
    """Build the command list for subprocess."""
    return [python_exe, str(script_path), *forwarded_args]


def run_subprocess(cmd: List[str], cwd: Path, dry_run: bool = False) -> int:
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
        process = subprocess.Popen(cmd, cwd=str(cwd))
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
        prog="python -m synrxn.main",
        description="SynRXN developer CLI (dataset builders).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build subcommand
    build_p = subparsers.add_parser("build", help="Run dataset builder scripts.")
    # builder flags (can be combined)
    build_p.add_argument("--aam", action="store_true", help="Run AAM dataset builder.")
    build_p.add_argument("--rbl", action="store_true", help="Run RBL dataset builder.")
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

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    main_path = Path(__file__).resolve()
    # main.py lives in synrxn/, repo root is one level up (or higher if we autodetect)
    repo_root = find_repo_root(main_path.parent)

    if args.command != "build":
        LOGGER.error("Only the 'build' subcommand is currently supported.")
        return 2

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

    if not targets:
        LOGGER.error(
            "No build target provided. Use one or more of "
            "--aam, --rbl, --classification, --synthesis, or --property."
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
        script_path = find_script_path(repo_root, args.script, target)
        if not script_path.exists():
            LOGGER.error("Target script for '%s' not found: %s", target, script_path)
            # Keep last non-zero, but continue with other targets
            overall_rc = overall_rc or 4
            continue

        cmd = build_command(python_exe, script_path, forwarded_for_builder)

        LOGGER.info(
            "Running target '%s': %s",
            target,
            " ".join(shlex.quote(c) for c in cmd),
        )
        LOGGER.debug("Working directory: %s", cwd)

        rc = run_subprocess(cmd, cwd=cwd, dry_run=args.dry_run)  # type: ignore
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
