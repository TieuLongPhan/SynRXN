#!/usr/bin/env bash
# lint.sh — run flake8 for the repository with sane defaults
set -euo pipefail

# repo root (script dir)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# default exclude patterns (comma-separated)
EXCLUDE=".git,__pycache__,.venv,venv,env,build,dist,.eggs,.mypy_cache,.pytest_cache,.cache,data,script,build_manifest.py"

# require flake8
if ! command -v flake8 >/dev/null 2>&1; then
  echo "flake8 is not installed. Install it with: pip install flake8" >&2
  exit 2
fi

# run flake8 on repository root; allow extra args to be passed through
flake8 "$ROOT" \
  --count \
  --statistics \
  --max-complexity=16 \
  --max-line-length=120 \
  --exclude="$EXCLUDE" \
  "$@"

exit $?
