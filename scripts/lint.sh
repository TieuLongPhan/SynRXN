#!/usr/bin/env bash
# Check ordering and launcher convention follow TieuLongPhan/SynKit scripts/lint.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT_DIR}"
PATHS=(synrxn Test scripts)
if (( $# )); then
    PATHS=("$@")
fi
status=0
"${PYTHON_BIN}" scripts/check_test_layout.py || status=1
"${PYTHON_BIN}" scripts/check_python_file_size.py "${PATHS[@]}" || status=1
"${PYTHON_BIN}" scripts/check_docstring_style.py "${PATHS[@]}" || status=1
"${PYTHON_BIN}" -m flake8 "${PATHS[@]}" --count --statistics || status=1
exit "${status}"
