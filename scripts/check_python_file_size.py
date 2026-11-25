"""Apply SynKit's 1,000-line limit after excluding Python docstrings.

Existing oversized modules have exact, non-growing limits in pyproject.toml.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"__pycache__", ".git", ".venv", "venv", "archive", ".cache"}
DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def python_files(paths: list[str]) -> list[Path]:
    files = set()
    for name in paths:
        path = Path(name)
        if not path.exists():
            raise ValueError(f"missing lint path: {name}")
        candidates = path.rglob("*.py") if path.is_dir() else [path]
        files.update(p.resolve() for p in candidates
                     if p.suffix == ".py" and not EXCLUDED.intersection(p.parts))
    return sorted(files)


def code_lines(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    excluded = set()
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, DOC_OWNERS) and ast.get_docstring(node) is not None:
            doc = node.body[0]
            excluded.update(range(doc.lineno, doc.end_lineno + 1))
    return len(source.splitlines()) - len(excluded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["synrxn", "Test", "scripts"])
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["synrxn"]["quality"]
    maximum = config["max-python-lines"]
    baseline = {ROOT / name: count for name, count in config["file-size-baseline"].items()}
    errors = []
    try:
        files = python_files(args.paths)
        for path in files:
            count = code_lines(path)
            limit = baseline.get(path, maximum)
            if count > limit:
                errors.append(f"{path}: {count} non-docstring lines; limit {limit}")
            if path in baseline and count <= maximum:
                errors.append(f"{path}: remove obsolete file-size baseline")
        # Only a full scan can determine whether a baseline file is missing.
        if all((ROOT / name).resolve() in {Path(p).resolve() for p in args.paths}
               for name in ("synrxn", "Test", "scripts")):
            errors.extend(f"{path}: remove missing file-size baseline" for path in baseline if not path.exists())
    except (ValueError, SyntaxError) as error:
        errors.append(str(error))
    for error in errors:
        print(error)
    if not errors:
        print("Python file sizes are within their limits.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
