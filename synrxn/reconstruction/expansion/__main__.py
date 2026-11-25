"""Acquire, build and inspect expansion source tables."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

from synrxn.reconstruction.expansion.catalog import ExpansionCatalog
from synrxn.reconstruction.paths import family_storage, physical_path


def main(argv=None):
    """Run the selected expansion operation."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["fetch"]:
        from synrxn.reconstruction.expansion.fetch import main as fetch

        return fetch(args[1:])
    if args[:1] == ["audit"]:
        from synrxn.reconstruction.expansion.audit_sources import main as audit

        return audit(args[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Data/expansion"))
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--repo", type=Path, default=Path.cwd())
    build.add_argument("--source", action="append")
    commands.add_parser("list")
    commands.add_parser("sources")
    commands.add_parser("verify")
    sample = commands.add_parser("sample")
    sample.add_argument("key")
    sample.add_argument("--task")
    sample.add_argument("--split", choices=["train", "validation", "test"])
    sample.add_argument("--limit", type=int, default=1)
    commands.add_parser("fetch", help="Fetch pinned expansion source files")
    commands.add_parser("audit", help="Refresh primary source metadata")
    args = parser.parse_args(args)
    if args.command in {"fetch", "audit"}:
        parser.error("--root applies to local table commands")
    if args.command == "build":
        from synrxn.reconstruction.expansion.build import build as run_build

        repo = args.repo.resolve()
        output = physical_path(args.root.resolve())
        if family_storage(repo) and output == physical_path("Data/expansion", repo):
            output = repo / ".cache/synrxn/expansion"
        result = run_build(repo, output, sources=args.source)
        print(json.dumps({"output": str(output), "tables": len(result["tables"]), "issues": len(result["issues"])}))
        return 0
    catalog = ExpansionCatalog(args.root)
    if args.command == "verify":
        results = catalog.verify()
        print(json.dumps(results, indent=2))
        return 0 if all(r["sha256_match"] and r["rows_match"] for r in results) else 1
    if args.command == "list":
        print(json.dumps(catalog.list(), indent=2))
    elif args.command == "sources":
        print(json.dumps(list(catalog.sources.values()), indent=2))
    elif args.command == "sample":
        if args.task:
            rows = catalog.task_view(args.key, args.task, args.split)
        else:
            rows = catalog.records(args.key, args.split)
        for row in itertools.islice(rows, args.limit):
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
