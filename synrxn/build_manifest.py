#!/usr/bin/env python3
from __future__ import annotations
import sys
import os
import re
import csv
import gzip
import io
import json
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, IO

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None  # type: ignore

try:
    import yaml  # optional (for nicer CFF output)

    HAVE_PYYAML = True
except Exception:
    HAVE_PYYAML = False

# optional dependencies for robust tabular handling
try:
    import pandas as pd  # type: ignore

    HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore
    HAVE_PANDAS = False

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_ZENODO_DOI = "10.5281/zenodo.17297723"


# -------------------------
# Utilities
# -------------------------
def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for ch in iter(lambda: fh.read(chunk), b""):
            h.update(ch)
    return h.hexdigest()


def mtime_iso(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts, timezone.utc).strftime(ISO_FMT)


def run(
    argv: List[str], cwd: Optional[Path] = None, timeout: int = 10
) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        out = proc.stdout.decode("utf8", "replace").strip()
        err = proc.stderr.decode("utf8", "replace").strip()
        return proc.returncode, out, err
    except Exception as e:
        return 1, "", str(e)


def find_git_root(start: Optional[Path] = None) -> Optional[Path]:
    start = (start or Path.cwd()).resolve()
    code, out, _ = run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    if code == 0 and out:
        try:
            return Path(out).resolve()
        except Exception:
            return None
    return None


def git_provenance(repo_root: Optional[Path]) -> Dict[str, Optional[str]]:
    prov = {"git_root": None, "commit": None, "branch": None, "remotes": None}
    root = find_git_root(repo_root)
    if not root:
        return prov
    prov["git_root"] = str(root)
    _, commit, _ = run(["git", "rev-parse", "HEAD"], cwd=root)
    prov["commit"] = commit or None
    c, branch, _ = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    prov["branch"] = branch if (c == 0 and branch and branch != "HEAD") else None
    _, remotes, _ = run(["git", "remote", "-v"], cwd=root)
    prov["remotes"] = remotes or None
    return prov


def load_pyproject(pyproject_path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not pyproject_path.exists() or not tomllib:
        return data
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)  # type: ignore
    return data


def extract_project_meta(pyproj: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    p = pyproj.get("project") or {}
    if p:
        meta["name"] = p.get("name")
        meta["version"] = p.get("version")
        meta["description"] = p.get("description")
        authors = p.get("authors") or []
        a_norm = []
        for a in authors:
            if isinstance(a, dict):
                nm = a.get("name") or ""
                em = a.get("email") or None
                if nm:
                    item = {"name": nm}
                    if em:
                        item["email"] = em
                    a_norm.append(item)
        meta["authors"] = a_norm
        meta["license"] = (
            (p.get("license") or {}).get("text")
            if isinstance(p.get("license"), dict)
            else p.get("license")
        )
    tp = pyproj.get("tool", {}).get("poetry") or {}
    if tp:
        meta.setdefault("name", tp.get("name"))
        meta.setdefault("version", tp.get("version"))
        meta.setdefault("description", tp.get("description"))
        authors = tp.get("authors") or []
        a_norm = meta.get("authors") or []
        for a in authors:
            s = str(a)
            if "<" in s and ">" in s:
                nm, rest = s.split("<", 1)
                em = rest.split(">", 1)[0].strip()
                a_norm.append({"name": nm.strip(), "email": em})
            else:
                a_norm.append({"name": s.strip()})
        if a_norm:
            meta["authors"] = a_norm
        meta.setdefault("license", tp.get("license"))
    return meta


# -------------------------
# Sidecar metadata loading
# -------------------------
def load_sidecar_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Search for top-level metadata files (metadata.yaml/json) and per-file sidecars.
    Returns a mapping: relative-path -> dict with keys like 'description', 'license'.
    """
    out: Dict[str, Dict[str, Any]] = {}

    # top-level metadata candidates
    candidates = [
        "metadata.yaml",
        "metadata.yml",
        "file_metadata.json",
        "metadata.json",
    ]
    for c in candidates:
        p = root / c
        if p.exists():
            try:
                if p.suffix in (".yaml", ".yml") and HAVE_PYYAML:
                    with p.open("r", encoding="utf8") as fh:
                        data = yaml.safe_load(fh) or {}
                else:
                    with p.open("r", encoding="utf8") as fh:
                        data = json.load(fh)
                if isinstance(data, dict):
                    # expected mapping of file-key -> metadata
                    for k, v in data.items():
                        if not isinstance(v, dict):
                            continue
                        out[str(k)] = v
            except Exception:
                continue

    # per-file sidecars: look for files named <file>.meta.json or <file>.meta.yaml
    for rootdir, _dirs, files in os.walk(str(root)):
        root_path = Path(rootdir)
        for fn in files:
            if (
                fn.endswith(".meta.json")
                or fn.endswith(".meta.yaml")
                or fn.endswith(".meta.yml")
            ):
                try:
                    meta_p = root_path / fn
                    with meta_p.open("r", encoding="utf8") as fh:
                        if meta_p.suffix == ".json":
                            data = json.load(fh)
                        else:
                            if HAVE_PYYAML:
                                data = yaml.safe_load(fh) or {}
                            else:
                                # naive yaml-ish fallback: try json load
                                try:
                                    data = json.load(fh)
                                except Exception:
                                    data = {}
                    # determine target key: strip .meta.* and relative to root
                    target_name = fn.rsplit(".meta", 1)[0]
                    rel = None
                    # try to find the corresponding file under root
                    cand = list(root_path.glob(target_name)) or []
                    # store under relative path (best-effort)
                    if cand:
                        rel = str(cand[0].relative_to(root).as_posix())
                    else:
                        rel = target_name
                    if isinstance(data, dict):
                        out[rel] = data
                except Exception:
                    continue
    return out


def _open_maybe_gz(p: Path) -> IO[bytes]:
    if str(p).endswith(".gz"):
        return gzip.open(p, "rb")
    return p.open("rb")


def detect_format_from_name(p: Path) -> str:
    name = p.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return "csv"
    if (
        name.endswith(".tsv")
        or name.endswith(".tsv.gz")
        or name.endswith(".txt")
        or name.endswith(".txt.gz")
    ):
        return "tsv"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".ndjson") or name.endswith(".jsonl"):
        return "ndjson"
    if name.endswith(".json"):
        return "json"
    return "binary"


def analyze_file(
    p: Path, count_rows: bool = True, rows_size_limit: int = 100 * 1024 * 1024
) -> Dict[str, Any]:
    """
    Return additional metadata for a file: format, mime, rows (int|None), columns (list|None).
    For performance reasons, row/column extraction is skipped if file size > rows_size_limit (in bytes)
    unless count_rows is False (then they are always skipped).
    """
    info: Dict[str, Any] = {"format": None, "mime": None, "rows": None, "columns": None}
    fmt = detect_format_from_name(p)
    info["format"] = fmt

    # small mime hint
    if fmt in ("csv", "tsv"):
        info["mime"] = "text/tabular"
    elif fmt in ("json", "ndjson"):
        info["mime"] = "application/json"
    elif fmt == "parquet":
        info["mime"] = "application/parquet"
    else:
        info["mime"] = None

    size = p.stat().st_size
    # if file is large and counting rows, skip by default
    if not count_rows or size > rows_size_limit:
        return info

    # CSV/TSV handling (gzip support)
    try:
        if fmt in ("csv", "tsv"):
            delim = ","
            if fmt == "tsv":
                delim = "\t"
            # open as text with correct newline handling
            bfh = _open_maybe_gz(p)
            with io.TextIOWrapper(
                bfh, encoding="utf8", errors="replace", newline=""
            ) as fh:
                # sniff delimiter if possible (may fail on weird inputs)
                sample = fh.read(8192)
                fh.seek(0)
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    delim = dialect.delimiter
                except Exception:
                    pass
                reader = csv.reader(fh, delimiter=delim)
                # attempt to read header as columns
                try:
                    first = next(reader)
                except StopIteration:
                    return info
                # determine if header-like: heuristics (no strictly numeric or small tokens)
                header_is_strings = any(not is_number(x) for x in first)
                if header_is_strings:
                    columns = [c.strip() for c in first]
                    info["columns"] = columns
                else:
                    info["columns"] = None
                # continue counting rows (count including header if header present)
                count = 1
                for _ in reader:
                    count += 1
                info["rows"] = count
                return info

        # NDJSON / JSON-lines
        if fmt in ("ndjson",):
            bfh = _open_maybe_gz(p)
            with io.TextIOWrapper(bfh, encoding="utf8", errors="replace") as fh:
                count = 0
                columns = None
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict):
                            if columns is None:
                                columns = list(obj.keys())
                        count += 1
                    except Exception:
                        # skip bad lines
                        continue
                info["rows"] = count
                info["columns"] = columns
                return info

        # JSON array (["obj", ...] or [{"k":v},...])
        if fmt == "json":
            # try to parse safely
            bfh = _open_maybe_gz(p)
            with io.TextIOWrapper(bfh, encoding="utf8", errors="replace") as fh:
                try:
                    data = json.load(fh)
                    if isinstance(data, list):
                        info["rows"] = len(data)
                        if len(data) > 0 and isinstance(data[0], dict):
                            info["columns"] = list(data[0].keys())
                    elif isinstance(data, dict):
                        info["rows"] = 1
                        info["columns"] = list(data.keys())
                    return info
                except Exception:
                    # fallback to streaming counting per-line objects
                    fh.seek(0)
                    count = 0
                    columns = None
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                            if isinstance(obj, dict) and columns is None:
                                columns = list(obj.keys())
                            count += 1
                        except Exception:
                            continue
                    if count > 0:
                        info["rows"] = count
                        info["columns"] = columns
                        return info
                    return info

        # Parquet (use pandas if available)
        if fmt == "parquet" and HAVE_PANDAS:
            try:
                # read only meta to avoid reading whole file
                df = pd.read_parquet(str(p), engine=None)
                info["rows"] = int(df.shape[0])
                info["columns"] = list(df.columns.astype(str))
                return info
            except Exception:
                # swallow and return best-effort
                return info

    except Exception:
        # never crash manifest build due to file analysis
        return info

    return info


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


# -------------------------
# Files scan & summary (enhanced)
# -------------------------
def scan_files(
    data_root: Path,
    follow_symlinks: bool = False,
    count_rows: bool = True,
    rows_size_limit: int = 100 * 1024 * 1024,
    sidecar: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    sidecar = sidecar or {}
    for root, _dirs, names in os.walk(str(data_root), followlinks=follow_symlinks):
        root_path = Path(root)
        for fn in sorted(names):
            p = root_path / fn
            try:
                if not p.is_file():
                    continue
                rel = p.relative_to(data_root).as_posix()
                entry: Dict[str, Any] = {
                    "key": rel,
                    "size": p.stat().st_size,
                    "sha256": sha256_file(p),
                    "mtime": mtime_iso(p),
                }
                # augment with analysis when reasonable
                analysis = analyze_file(
                    p, count_rows=count_rows, rows_size_limit=rows_size_limit
                )
                # copy non-null analysis fields
                for k, v in analysis.items():
                    if v is not None:
                        entry[k] = v

                # inject sidecar metadata (description/license) if present
                # sidecar keys may be full relative path or just filename keys
                sc = sidecar.get(rel) or sidecar.get(p.name)
                if sc and isinstance(sc, dict):
                    if "description" in sc:
                        entry["description"] = sc["description"]
                    if "license" in sc:
                        entry["license"] = sc["license"]

                files.append(entry)
            except Exception:
                continue
    files.sort(key=lambda d: d["key"])
    return files


def summarize(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(int(f.get("size", 0)) for f in files)
    by_dir: Dict[str, Dict[str, int]] = {}
    for f in files:
        key = f["key"]
        top = key.split("/", 1)[0] if "/" in key else "."
        ent = by_dir.setdefault(top, {"count": 0, "bytes": 0})
        ent["count"] += 1
        ent["bytes"] += int(f.get("size", 0))
    return {"file_count": len(files), "total_bytes": total, "by_dir": by_dir}


def iso_date(iso_or_date: Optional[str]) -> Optional[str]:
    if not iso_or_date:
        return None
    try:
        dt = datetime.fromisoformat(iso_or_date.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", iso_or_date)
        return m.group(1) if m else None


def split_name(full: str) -> Dict[str, str]:
    s = full.strip()
    if not s:
        return {"given-names": "", "family-names": ""}
    if "," in s:
        fam, giv = [p.strip() for p in s.split(",", 1)]
        return {"given-names": giv, "family-names": fam}
    toks = s.split()
    if len(toks) == 1:
        return {"given-names": "", "family-names": toks[0]}
    return {"given-names": " ".join(toks[:-1]), "family-names": toks[-1]}


def normalize_license(lic: Optional[str]) -> Optional[str]:
    return str(lic).strip().replace(" ", "-") if lic else None


def first_remote_url(remotes: Optional[str]) -> Optional[str]:
    if not remotes:
        return None
    m = re.search(r"(https?://[^\s]+|git@[^)\s]+)", remotes)
    return m.group(1) if m else None


def authors_to_cff(auth: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for a in auth:
        if isinstance(a, dict) and (a.get("given-names") or a.get("family-names")):
            ent = {}
            if a.get("given-names"):
                ent["given-names"] = a["given-names"]
            if a.get("family-names"):
                ent["family-names"] = a["family-names"]
            if a.get("email"):
                ent["email"] = a["email"]
            if a.get("affiliation"):
                ent["affiliation"] = a["affiliation"]
            out.append(ent)
            continue
        name = (a.get("name") if isinstance(a, dict) else str(a)) or ""
        email = a.get("email") if isinstance(a, dict) else None
        sp = split_name(name)
        ent = {}
        if sp.get("given-names"):
            ent["given-names"] = sp["given-names"]
        if sp.get("family-names"):
            ent["family-names"] = sp["family-names"]
        if email:
            ent["email"] = email
        out.append(ent)
    return out


def dump_yaml_manual(data: Dict[str, Any], outpath: Path) -> None:
    def esc(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        if re.search(r"[:\-\[\]\{\},#&*!|>\'\"%@`]", s) or s.strip() != s or "\n" in s:
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{s}"'
        return s

    order = [
        "cff-version",
        "title",
        "version",
        "type",
        "message",
        "abstract",
        "date-released",
        "doi",
        "url",
        "license",
        "repository-code",
        "authors",
        "identifiers",
    ]
    lines: List[str] = []
    for k in order:
        if k not in data:
            continue
        v = data[k]
        if k in ("authors", "identifiers") and isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append("  -")
                for kk, vv in item.items():
                    lines.append(f"    {kk}: {esc(vv)}")
        else:
            lines.append(f"{k}: {esc(v)}")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf8")


def build_cff_from_manifest(
    manifest: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    ds = manifest.get("dataset", {}) or {}
    prov = manifest.get("provenance", {}) or {}

    cff: Dict[str, Any] = {"cff-version": "1.2.0"}
    cff["title"] = overrides.get("title") or ds.get("title") or "SynRXN dataset"
    v = overrides.get("version") or ds.get("version")
    if v:
        cff["version"] = str(v)

    cff["type"] = overrides.get("type") or "dataset"
    cff["message"] = (
        overrides.get("message")
        or "If you use this work, please cite it using this file."
    )
    if ds.get("description"):
        cff["abstract"] = ds["description"]

    doi = overrides.get("doi") or ds.get("doi")
    if doi:
        cff["doi"] = str(doi)

    if overrides.get("url"):
        cff["url"] = overrides["url"]
    elif doi and not overrides.get("url"):
        cff["url"] = f"https://doi.org/{doi}"

    lic = overrides.get("license") or ds.get("license")
    lic = normalize_license(lic)
    if lic:
        cff["license"] = lic

    repo = overrides.get("repo_url") or first_remote_url(prov.get("remotes"))
    if repo:
        cff["repository-code"] = repo

    date_rel = overrides.get("date_released") or manifest.get("generated_at")
    if date_rel:
        dr = iso_date(date_rel)
        if dr:
            cff["date-released"] = dr

    authors = ds.get("authors") or []
    if authors:
        cff["authors"] = authors_to_cff(authors)

    idents: List[Dict[str, str]] = []
    if prov.get("commit"):
        idents.append({"type": "commit", "value": prov["commit"]})
    if doi:
        idents.append({"type": "doi", "value": str(doi)})
    if overrides.get("swhid"):
        idents.append({"type": "swh", "value": str(overrides["swhid"])})
    if idents:
        cff["identifiers"] = idents
    return cff


# -------------------------
# Code manifest generator
# -------------------------
def generate_code_manifest(
    repo_root: Path,
    subdirs: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a code manifest dict for files under repo_root/subdirs.
    Returns a mapping:
    {
      "generated_at": ...,
      "repo_root": "...",
      "subdirs": [...],
      "entries": [{ "key": "synrxn/...", "size": ..., "sha256": "...", "mtime": "..." }, ...],
      "count": N
    }
    """
    subdirs = subdirs or ["synrxn"]
    exclude_patterns = exclude_patterns or []
    repo_root = repo_root.resolve()
    entries: List[Dict[str, Any]] = []
    for sd in subdirs:
        base = repo_root / sd
        if not base.exists():
            continue
        for root, _dirs, files in os.walk(str(base)):
            rootp = Path(root)
            for fn in sorted(files):
                path = rootp / fn
                rel = path.relative_to(repo_root).as_posix()
                # skip unwanted patterns
                if any(re.search(pat, rel) for pat in exclude_patterns):
                    continue
                try:
                    if not path.is_file():
                        continue
                    entry = {
                        "key": rel,
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "mtime": mtime_iso(path),
                    }
                    entries.append(entry)
                except Exception:
                    # non-fatal: skip problematic files
                    continue
    entries.sort(key=lambda e: e["key"])
    code_manifest = {
        "generated_at": now_iso_utc(),
        "repo_root": str(repo_root),
        "subdirs": subdirs,
        "entries": entries,
        "count": len(entries),
    }
    return code_manifest


# -------------------------
# Build manifest (updated)
# -------------------------
def build_manifest(
    data_dir: Path,
    meta: Dict[str, Any],
    follow_symlinks: bool = False,
    doi: Optional[str] = None,
    license_str: Optional[str] = None,
    count_rows: bool = True,
    rows_size_limit: int = 100 * 1024 * 1024,
    include_tests: bool = False,
    include_code_archive: bool = False,
    include_code_manifest: bool = True,
    repo_root_hint: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build manifest for the dataset under `data_dir`.

    - include_tests: when True include top-level Test/ files found under the Data root.
    - include_code_archive: when True create a git archive (synrxn/ or whole repo) and add it as a dataset file.
    - include_code_manifest: when True compute per-file sha256 for synrxn/* and embed under manifest['code'].
    - repo_root_hint: optional Path to repository root to prefer when creating archive/code manifest.
    """
    generated_at = now_iso_utc()
    root = data_dir
    if (root / "Data").is_dir():
        root = root / "Data"
    if not root.exists():
        raise FileNotFoundError(f"data-dir not found: {root}")

    # load sidecar metadata map (optional)
    sidecar = load_sidecar_metadata(root)

    # scan everything under Data (we will filter to allowed prefixes below)
    files = scan_files(
        root,
        follow_symlinks=follow_symlinks,
        count_rows=count_rows,
        rows_size_limit=rows_size_limit,
        sidecar=sidecar,
    )

    # Filter to allowed Data subfolders by default
    allowed = {"rbl", "aam", "classification", "property", "synthesis"}
    filtered: List[Dict[str, Any]] = []
    for f in files:
        top = f["key"].split("/", 1)[0] if "/" in f["key"] else f["key"]
        # keep allowed data subdirs
        if top in allowed:
            filtered.append(f)
            continue
        # optionally include Test/*
        if include_tests and top == "Test":
            filtered.append(f)
            continue
        # allow some top-level helper files (README, CITATION, top-level metadata)
        if top.lower() in {"readme.md", "readme", "citation.cff", "citation", "."}:
            filtered.append(f)
            continue
        # otherwise skip
    files = filtered

    stats = summarize(files)

    # provenance from git (try data root, then repo_root_hint)
    prov = git_provenance(repo_root=root)
    if not prov.get("git_root") and repo_root_hint:
        prov = git_provenance(repo_root_hint)

    # Prepare code container for embedding (if requested)
    code_section: Dict[str, Any] = {}

    # Optionally create a git archive for synrxn/ (or whole repo) and include it
    if include_code_archive:
        gitroot = (
            find_git_root(root)
            or find_git_root(repo_root_hint)
            or find_git_root(Path.cwd())
        )
        if gitroot:
            # determine commit
            _, commit_out, _ = run(["git", "rev-parse", "HEAD"], cwd=gitroot)
            commit = commit_out.strip() if commit_out else "HEAD"
            short = commit[:7] if commit and commit != "HEAD" else "HEAD"
            # name archive
            archive_name = f"synrxn-code-{short}.tar.gz"
            archive_path = gitroot / archive_name
            # decide whether to archive only synrxn/ or whole repo
            synrxn_dir = gitroot / "synrxn"
            try:
                if synrxn_dir.exists():
                    cmd = [
                        "git",
                        "archive",
                        "--format=tar.gz",
                        "-o",
                        str(archive_path),
                        commit,
                        "synrxn",
                    ]
                else:
                    # no synrxn/ dir — archive whole repo at commit
                    cmd = [
                        "git",
                        "archive",
                        "--format=tar.gz",
                        "-o",
                        str(archive_path),
                        commit,
                        "HEAD",
                    ]
                code, out, err = run(cmd, cwd=gitroot, timeout=60)
                if code == 0 and archive_path.exists():
                    try:
                        sh = sha256_file(archive_path)
                        archive_entry = {
                            "key": f"code/{archive_name}",
                            "size": archive_path.stat().st_size,
                            "sha256": sh,
                            "mtime": mtime_iso(archive_path),
                            "description": "git archive of repository (synrxn/ or whole repo) at commit "
                            + (commit or "HEAD"),
                            "format": "tar.gz",
                            "mime": "application/gzip",
                        }
                        files.append(archive_entry)
                        # also expose as code section metadata
                        code_section["archive"] = {
                            "path": archive_entry["key"],
                            "size": archive_entry["size"],
                            "sha256": archive_entry["sha256"],
                            "commit": commit,
                        }
                    except Exception:
                        # non-fatal: continue without archive metadata
                        pass
            except Exception:
                # non-fatal: continue without adding archive
                pass

    # Optionally generate per-file code manifest and embed it under 'code' in manifest
    if include_code_manifest:
        gitroot = (
            find_git_root(root)
            or find_git_root(repo_root_hint)
            or find_git_root(Path.cwd())
        )
        if gitroot:
            cm = generate_code_manifest(
                repo_root=gitroot,
                subdirs=["synrxn"],
                exclude_patterns=[r"\.pyc$", r"__pycache__"],
            )
            # embed code manifest (in-memory)
            code_section["manifest"] = cm

    dataset = {
        "title": meta.get("title"),
        "version": meta.get("version"),
        "description": meta.get("description"),
        "license": license_str,
        "doi": doi,
        "authors": meta.get("authors") or [],
        "files": files,
        "summary": stats,
    }
    dataset = {k: v for k, v in dataset.items() if v not in (None, "", [])}

    man: Dict[str, Any] = {
        "generated_at": generated_at,
        "dataset": dataset,
        "provenance": prov,
        "project": {
            "name": meta.get("name"),
            "pyproject": meta.get("_pyproject_path"),
        },
    }

    if code_section:
        man["code"] = code_section

    return man


# -------------------------
# CLI
# -------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build manifest.json and CITATION.cff (CITATION always generated)."
    )
    p.add_argument(
        "-d", "--data-dir", default="Data", help="Directory to scan (default: Data)"
    )
    p.add_argument(
        "-m", "--manifest-output", default="manifest.json", help="Output manifest path"
    )
    p.add_argument(
        "-c",
        "--citation-output",
        default="CITATION.cff",
        help="CITATION.cff path (always written)",
    )
    p.add_argument(
        "--title", help="Dataset title (default: from pyproject name or SynRXN dataset)"
    )
    p.add_argument("--version", help="Override version (default: from pyproject)")
    p.add_argument(
        "--description", help="Override description (default: from pyproject)"
    )
    p.add_argument(
        "--author", action="append", help="Extra author 'Name <email>' (can repeat)"
    )
    p.add_argument(
        "--license",
        dest="license_str",
        help="SPDX id or human license text (e.g. CC-BY-4.0)",
    )
    # Default Zenodo DOI baked in; user can override if they want.
    p.add_argument(
        "--doi",
        default=DEFAULT_ZENODO_DOI,
        help=f"Zenodo DOI to include (default: {DEFAULT_ZENODO_DOI})",
    )
    p.add_argument("--repo-url", help="Override repository-code URL")
    p.add_argument(
        "--url", help="Override project URL for CFF (e.g., https://doi.org/...)"
    )
    p.add_argument("--swhid", help="Add SWHID to CFF identifiers")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    # new options for row counting
    p.add_argument(
        "--count-rows",
        dest="count_rows",
        action="store_true",
        default=True,
        help="Attempt to count rows and list columns for tabular files (default: True)",
    )
    p.add_argument(
        "--no-count-rows",
        dest="count_rows",
        action="store_false",
        help="Do not attempt to count rows / columns (faster)",
    )
    p.add_argument(
        "--rows-size-limit",
        type=int,
        default=100 * 1024 * 1024,
        help="Maximum file size (bytes) to scan for rows/columns (default 100MB). Set 0 for no limit.",
    )

    # new options for included paths and code archive/manifest
    p.add_argument(
        "--include-tests",
        action="store_true",
        help="Also include Test/* paths in the manifest (default: False)",
    )
    p.add_argument(
        "--include-code-archive",
        action="store_true",
        help="Create a git-archive of repository (synrxn/* or whole repo) and include it in manifest files.",
    )
    # include code manifest by default; add inverse flag to disable
    p.add_argument(
        "--include-code-manifest",
        dest="include_code_manifest",
        action="store_true",
        help="Embed a per-file code manifest for synrxn/* inside manifest.json (default: True).",
    )
    p.add_argument(
        "--no-include-code-manifest",
        dest="include_code_manifest",
        action="store_false",
        help="Do not embed per-file code manifest for synrxn/* inside manifest.json.",
    )
    # set default True
    p.set_defaults(include_code_manifest=True)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_out = Path(args.manifest_output).expanduser().resolve()
    citation_out = Path(args.citation_output).expanduser().resolve()

    repo_root = find_git_root(data_dir) or data_dir
    pyproject_path = repo_root / "pyproject.toml"
    pyproj = load_pyproject(pyproject_path)
    proj_meta = extract_project_meta(pyproj)
    proj_meta["_pyproject_path"] = (
        str(pyproject_path) if pyproject_path.exists() else None
    )

    meta: Dict[str, Any] = {}
    meta["name"] = proj_meta.get("name") or "SynRXN"
    # version precedence: CLI override > pyproject version
    meta["version"] = args.version or proj_meta.get("version")
    meta["description"] = args.description or proj_meta.get("description")
    authors = proj_meta.get("authors") or []
    extra = []
    for s in args.author or []:
        s = str(s).strip()
        if "<" in s and ">" in s:
            nm, rest = s.split("<", 1)
            em = rest.split(">", 1)[0].strip()
            extra.append({"name": nm.strip(), "email": em})
        else:
            extra.append({"name": s})
    if extra:
        authors = (authors or []) + extra
    meta["authors"] = authors
    meta["title"] = (
        args.title
        or f"{meta['name']}: A Benchmarking Framework and Open Data Repository for Computer-Aided Synthesis Planning"
    )

    # If no version found, warn (but still proceed); CITATION will include whatever version we can find.
    if not meta.get("version"):
        print(
            "[warning] version not found in pyproject.toml and --version not provided; CITATION.cff will omit version.",
            file=sys.stderr,
        )

    try:
        manifest = build_manifest(
            data_dir=data_dir,
            meta=meta,
            follow_symlinks=args.follow_symlinks,
            doi=args.doi,
            license_str=args.license_str,
            count_rows=args.count_rows,
            rows_size_limit=(
                args.rows_size_limit if args.rows_size_limit > 0 else sys.maxsize
            ),
            include_tests=args.include_tests,
            include_code_archive=args.include_code_archive,
            include_code_manifest=args.include_code_manifest,
            repo_root_hint=repo_root,
        )
    except Exception as exc:
        print(f"[manifest] error: {exc}", file=sys.stderr)
        return 2

    try:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
        )
    except Exception as exc:
        print(f"[manifest] write failed: {exc} (path: {manifest_out})", file=sys.stderr)
        return 3

    if args.verbose:
        ds = manifest["dataset"]
        print(f"[manifest] wrote {manifest_out}")
        print(
            f"  files={ds['summary']['file_count']} bytes={ds['summary']['total_bytes']} version={ds.get('version')}"
        )
        if manifest.get("code"):
            cs = manifest["code"]
            if "archive" in cs:
                print(
                    f"  code archive: {cs['archive'].get('path')} sha256={cs['archive'].get('sha256')}"
                )
            if "manifest" in cs:
                print(f"  code manifest entries: {cs['manifest'].get('count')}")

    # CITATION.cff: ALWAYS generated
    overrides = {
        "title": meta.get("title"),
        "version": meta.get("version"),
        "doi": args.doi,
        "license": args.license_str,
        "repo_url": args.repo_url,
        "url": args.url or (f"https://doi.org/{args.doi}" if args.doi else None),
        "date_released": manifest.get("generated_at"),
        "swhid": args.swhid,
    }
    try:
        cff = build_cff_from_manifest(manifest, overrides)
    except Exception as exc:
        print(f"[citation] build failed: {exc}", file=sys.stderr)
        return 4
    try:
        citation_out.parent.mkdir(parents=True, exist_ok=True)
        if HAVE_PYYAML:
            with citation_out.open("w", encoding="utf8") as fh:
                yaml.safe_dump(cff, fh, sort_keys=False, allow_unicode=True)  # type: ignore
        else:
            dump_yaml_manual(cff, citation_out)
    except Exception as exc:
        print(f"[citation] write failed: {exc} (path: {citation_out})", file=sys.stderr)
        return 5
    if args.verbose:
        print(f"[citation] wrote {citation_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
