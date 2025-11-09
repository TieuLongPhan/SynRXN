from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import re
import json
import hashlib


def normalize_version(v: str) -> str:
    """
    Normalize a version string by stripping a leading 'v' or 'V'.

    :param v: Version string (e.g., "v0.0.5", "0.0.5").
    :return: Normalized version (e.g., "0.0.5").
    """
    v = str(v).strip()
    if v.lower().startswith("v"):
        v = v[1:]
    return v


def parse_checksum_field(checksum_field: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a Zenodo checksum field like "sha256:<HEX>".

    :param checksum_field: Checksum string from Zenodo metadata.
    :return: (algorithm, hex) or (None, None) if not parseable.
    """
    if not checksum_field:
        return None, None
    m = re.match(
        r"^(md5|sha1|sha224|sha256|sha384|sha512):([0-9A-Fa-f]+)$",
        checksum_field.strip(),
    )
    if not m:
        return None, None
    return m.group(1), m.group(2)


def save_json_silent(path: Optional[Path], data: Dict) -> None:
    """
    Save a mapping to JSON, ignoring write errors.

    :param path: Destination path (may be None).
    :param data: Mapping to serialize.
    """
    if not path:
        return
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def load_json_silent(path: Optional[Path]) -> Dict:
    """
    Load JSON from file, returning {} on any failure.

    :param path: Path to JSON file (may be None).
    :return: Dict or {}.
    """
    if not path:
        return {}
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def sha256_hex(s: str) -> str:
    """
    Compute a short SHA-256 hex digest for a string.

    :param s: Input string.
    :return: Hex digest (lowercase).
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
