"""Check that Test mirrors the module and package directories in synrxn."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path) -> list[str]:
    package, tests = root / "synrxn", root / "Test"
    errors = []
    if not (package / "__init__.py").is_file() or not tests.is_dir():
        return ["expected synrxn/__init__.py and Test/"]
    packages = {p.parent.relative_to(package) for p in package.rglob("__init__.py")}
    modules = sorted(tests.rglob("test_*.py"))
    for relative in sorted(packages):
        if not any(p.is_relative_to(tests / relative) for p in modules):
            errors.append(f"Test/{relative}: missing tests for synrxn/{relative}")
    for path in modules:
        relative = path.parent.relative_to(tests)
        if not any((package / relative).glob("*.py")):
            errors.append(f"{path.relative_to(root)}: no matching synrxn module directory")
    for path in sorted(tests.rglob("*.py")):
        if path.name.startswith("test_") or path.name in {"conftest.py", "__init__.py"}:
            continue
        errors.append(f"{path.relative_to(root)}: move non-test code to synrxn/")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = check(args.root.resolve())
    for error in errors:
        print(error)
    if not errors:
        print("Test package layout matches synrxn.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
