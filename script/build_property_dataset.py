#!/usr/bin/env python3
"""
build_property_dataset.py

Robust builder for property datasets (complete script).
 - Handles remote raw GitHub file collections, ZIP, TAR.GZ streams, plain CSV/JSON/XLSX and local folders.
 - Selective streaming extraction for large chemprop Zenodo tarball: only regular FILE matches
   count toward early-stop (fixes directory-only-match issue).
 - Removes temporary extraction dir after successful load unless CHEMPROP_EXTRACT_DIR is set.
 - Summary includes columns_preview and columns_full (semicolon-separated).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# ---------------------------
# DEFAULT CONFIG
# ---------------------------
DEFAULT_CONFIG: Dict[str, Dict[str, str]] = {
    "b97xd3": {
        "src": "https://zenodo.org/records/3715478/files/b97d3.csv?download=1",
        "out": "Data/property/b97xd3.csv.gz",
    },
    "snar": {
        "src": "https://www.rsc.org/suppdata/d0/sc/d0sc04896h/d0sc04896h2.zip",
        "out": "Data/property/snar.csv.gz",
    },
    "e2sn2": {
        "src": "https://raw.githubusercontent.com/hesther/reactiondatabase/refs/heads/main/data/e2sn2.csv",
        "out": "Data/property/e2sn2.csv.gz",
    },
    "rad6re": {
        "src": "https://github.com/hesther/reactiondatabase/raw/refs/heads/main/data/rad6re.csv",
        "out": "Data/property/rad6re.csv.gz",
    },
    "lograte": {
        "src": "https://raw.githubusercontent.com/hesther/reactiondatabase/refs/heads/main/data/lograte.csv",
        "out": "Data/property/lograte.csv.gz",
    },
    "phosphatase": {
        "src": "https://github.com/hesther/reactiondatabase/raw/refs/heads/main/data/phosphatase.csv",
        "out": "Data/property/phosphatase.csv.gz",
    },
    "phosphatase_onehotenzyme_aux": {
        "src": "https://github.com/hesther/reactiondatabase/raw/refs/heads/main/data/phosphatase_onehotenzyme.csv",
        "out": "Data/property/_aux/phosphatase_onehotenzyme.csv.gz",
    },
    "chemprop_bundle": {
        "src": "https://zenodo.org/records/10078142/files/data.tar.gz?download=1",
        "out": "Data/property/_bundles/chemprop_data.tar.gz",
    },
    "chemprop_e2": {
        "src": "chemprop_zenodo/data/barriers_e2",
        "out": "Data/property/e2.csv.gz",
    },
    "chemprop_sn2": {
        "src": "chemprop_zenodo/data/barriers_sn2",
        "out": "Data/property/sn2.csv.gz",
    },
    "chemprop_rdb7": {
        "src": "chemprop_zenodo/data/barriers_rdb7",
        "out": "Data/property/rdb7.csv.gz",
    },
    "chemprop_cycloadd": {
        "src": "chemprop_zenodo/data/barriers_cycloadd",
        "out": "Data/property/cycloadd.csv.gz",
    },
    "chemprop_rgd1_local": {
        "src": "chemprop_zenodo/data/barriers_rgd1",
        "out": "Data/property/rgd1.csv.gz",
    },
    "suzuki_miyaura": {
        "src": "https://raw.githubusercontent.com/reymond-group/drfp/main/data/Suzuki-Miyaura/random_splits",
        "out": "Data/property/suzuki_miyaura.csv.gz",
    },
    "uspto_yields_above": {
        "src": "https://github.com/reymond-group/drfp/raw/refs/heads/main/data/uspto_yields_above.csv",
        "out": "Data/property/_aux/uspto_yields_above.csv.gz",
    },
    "uspto_yields_below": {
        "src": "https://github.com/reymond-group/drfp/raw/refs/heads/main/data/uspto_yields_below.csv",
        "out": "Data/property/_aux/uspto_yields_below.csv.gz",
    },
    "uspto_yield": {
        "src": "COMBINE(uspto_yields_above,uspto_yields_below)",
        "out": "Data/property/uspto_yield.csv.gz",
    },
}


# ---------------------------
# Utilities
# ---------------------------
def ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required but not importable") from e
    return pd


def http_get_bytes(
    url: str, timeout: int = 60, headers: Optional[Dict[str, str]] = None
) -> Tuple[bytes, int]:
    import requests

    hdrs = headers or {}
    resp = requests.get(url, timeout=timeout, headers=hdrs)
    resp.raise_for_status()
    return resp.content, resp.status_code


def download_to_bytes(
    url: str, *, timeout: int = 120, headers: Optional[Dict[str, str]] = None
) -> Tuple[bytes, int]:
    return http_get_bytes(url, timeout=timeout, headers=headers)


def save_df_gz(df, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")


# ---------------------------
# Token normalization & streamer
# ---------------------------
def _normalize_token(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "_", s.lower())


def stream_extract_selected_from_targz_safe(
    url: str,
    targets: List[str],
    dest_dir: str,
    *,
    timeout: int = 300,
    max_no_progress: int = 200000,
    min_matches_per_target: int = 1,
    verbose: bool = True,
) -> Dict[str, List[Path]]:
    """
    Stream a remote tar.gz and extract only members whose normalized path or basename
    matches any token in targets. Only **regular files** are counted toward the
    early-stop condition (min_matches_per_target). Directories are created for
    diagnostics but do not satisfy the file-match count.
    """
    import requests as _req

    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    norm_targets = [_normalize_token(t) for t in targets]
    target_map = {nt: t for nt, t in zip(norm_targets, targets)}
    extracted_by_target: Dict[str, List[Path]] = {nt: [] for nt in norm_targets}
    file_matches_by_target: Dict[str, int] = {nt: 0 for nt in norm_targets}
    seen_members: Set[str] = set()
    no_progress = 0
    total = 0
    first_matches: Dict[str, List[str]] = {nt: [] for nt in norm_targets}

    if verbose:
        print(f"[chemprop] stream_extract: url={url} dest={dest_dir} tokens={targets}")

    with _req.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                total += 1
                if member is None or not getattr(member, "name", None):
                    continue
                name = member.name
                if name in seen_members:
                    no_progress += 1
                    if no_progress >= max_no_progress:
                        if verbose:
                            print(
                                f"[chemprop] no progress for {max_no_progress} members -> stopping early"
                            )
                        break
                    continue
                seen_members.add(name)
                no_progress += 1

                name_norm = _normalize_token(name)
                base_norm = _normalize_token(Path(name).name)
                matched = [
                    tok
                    for tok in norm_targets
                    if (tok in name_norm or tok in base_norm)
                ]

                if not matched:
                    if verbose and total % 2000 == 0:
                        print(
                            f"[chemprop] scanned {total} entries (no token match yet)..."
                        )
                    if no_progress >= max_no_progress:
                        if verbose:
                            print(
                                f"[chemprop] no progress >= {max_no_progress} -> stopping early"
                            )
                        break
                    continue

                # match(es) found
                no_progress = 0
                if verbose:
                    print(f"[chemprop] matched member: {name} -> tokens {matched}")

                target_path = dest_dir_path / name
                target_resolved = target_path.resolve(strict=False)
                dest_resolved = dest_dir_path.resolve()
                if not str(target_resolved).startswith(str(dest_resolved)):
                    if verbose:
                        print(f"[chemprop] skip unsafe member path: {name}")
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # directories are recorded but DO NOT increment file match counters
                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    for tok in matched:
                        extracted_by_target[tok].append(target_path.resolve())
                        if len(first_matches[tok]) < 5:
                            first_matches[tok].append(name)
                    continue

                # regular files: extract and count
                if member.isreg():
                    f = tar.extractfile(member)
                    if f is None:
                        if verbose:
                            print(
                                f"[chemprop] warning: could not extract member {name}"
                            )
                        continue
                    with open(target_path, "wb") as outfh:
                        while True:
                            chunk = f.read(64 * 1024)
                            if not chunk:
                                break
                            outfh.write(chunk)
                    try:
                        os.chmod(target_path, member.mode)
                    except Exception:
                        pass
                    for tok in matched:
                        extracted_by_target[tok].append(target_path.resolve())
                        file_matches_by_target[tok] += 1
                        if len(first_matches[tok]) < 5:
                            first_matches[tok].append(name)
                else:
                    # other types: create small marker and record
                    with open(target_path, "w", encoding="utf-8") as outfh:
                        outfh.write(f"# skipped special member: {name}\n")
                    for tok in matched:
                        extracted_by_target[tok].append(target_path.resolve())
                        if len(first_matches[tok]) < 5:
                            first_matches[tok].append(name)

                # early exit: require file matches (regular files) for each token
                if all(
                    file_matches_by_target[tok] >= min_matches_per_target
                    for tok in norm_targets
                ):
                    if verbose:
                        print(
                            f"[chemprop] satisfied min_matches_per_target={min_matches_per_target} (file matches), stopping early"
                        )
                    break

    if verbose:
        print(f"[chemprop] streaming finished; scanned members: {total}")
        for nt in norm_targets:
            print(
                f"[chemprop] token '{target_map[nt]}' -> extracted {len(extracted_by_target[nt])} items "
                f"(files={file_matches_by_target[nt]}); sample matches: {first_matches[nt]}"
            )
    return extracted_by_target


# ---------------------------
# ZIP parsing (robust)
# ---------------------------
def parse_zip_bytes_to_dataframe(
    zip_bytes: bytes, target_basename: Optional[str] = None
):
    pd = ensure_pandas()
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = z.namelist()

    candidate = None
    if target_basename:
        tb = target_basename.lower()
        for n in names:
            if tb == os.path.basename(n).lower():
                candidate = n
                break
        if candidate is None:
            for n in names:
                if tb in os.path.basename(n).lower():
                    candidate = n
                    break
        if candidate is None:
            for n in names:
                if tb in n.lower():
                    candidate = n
                    break

    if candidate is None:
        for ext in (".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json"):
            for n in names:
                if n.lower().endswith(ext):
                    candidate = n
                    break
            if candidate:
                break

    if candidate is None:
        sample = names[:40]
        raise FileNotFoundError(f"No CSV/XLSX/JSON file found in ZIP. Sample: {sample}")

    raw_bytes = z.read(candidate)

    encs = ["cp1252", "latin1", "iso-8859-1", "utf-8", "utf-16"]
    try:
        import chardet  # type: ignore

        det = chardet.detect(raw_bytes[:200_000])
        enc = det.get("encoding")
        if enc and enc not in encs:
            encs = [enc] + encs
    except Exception:
        pass

    delim = ","
    for enc in encs:
        try:
            sample_text = raw_bytes[:200_000].decode(enc, errors="replace")
            sniff = csv.Sniffer()
            delim = sniff.sniff(sample_text).delimiter
            break
        except Exception:
            continue

    last_exc = None
    for enc in encs:
        try:
            bio = io.BytesIO(raw_bytes)
            try:
                return pd.read_csv(
                    bio, encoding=enc, delimiter=delim, engine="c", low_memory=False
                )
            except Exception:
                bio = io.BytesIO(raw_bytes)
                return pd.read_csv(
                    bio,
                    encoding=enc,
                    delimiter=delim,
                    engine="python",
                    low_memory=False,
                )
        except Exception as e:
            last_exc = e
            continue

    text = raw_bytes.decode("utf-8", errors="replace")
    try:
        return pd.read_csv(io.StringIO(text), delimiter=delim)
    except Exception as final_e:
        raise RuntimeError(
            f"Failed to parse CSV in ZIP. Last: {last_exc!r}; final: {final_e!r}"
        ) from final_e


# ---------------------------
# bytes -> DataFrame
# ---------------------------
def read_bytes_as_dataframe(b: bytes, src_hint: Optional[str] = None):
    pd = ensure_pandas()
    try:
        if zipfile.is_zipfile(io.BytesIO(b)):
            tb = None
            if src_hint and "snar" in (src_hint or "").lower():
                tb = "SNAR_reaction_dataset_SI.csv"
            return parse_zip_bytes_to_dataframe(b, tb)
    except Exception:
        pass

    try:
        bio = io.BytesIO(b)
        bio.seek(0)
        return pd.read_csv(bio, compression="infer", low_memory=False)
    except Exception:
        pass

    try:
        text = b.decode("utf-8")
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        if isinstance(parsed, dict):
            return pd.DataFrame(parsed)
    except Exception:
        pass

    try:
        bio = io.BytesIO(b)
        bio.seek(0)
        return pd.read_excel(bio)
    except Exception:
        pass

    raise RuntimeError(f"Unable to parse bytes to DataFrame (hint: {src_hint or ''})")


# ---------------------------
# GitHub listing helper
# ---------------------------
def github_raw_to_api(url: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.match(
        r"https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)$", url
    )
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/raw/([^/]+)/(.*)$", url)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/raw/refs/heads/([^/]+)/(.*)$", url
    )
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return None


def list_github_dir(url: str) -> List[str]:
    parsed = github_raw_to_api(url)
    if not parsed:
        return []
    owner, repo, branch, path = parsed
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    )
    headers: Dict[str, str] = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    import requests

    resp = requests.get(api_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        logger.warning("GitHub API list failed %s -> %d", api_url, resp.status_code)
        return []
    items = resp.json()
    files: List[str] = []
    for it in items:
        if it.get("type") == "file":
            ru = it.get("download_url")
            if ru:
                files.append(ru)
    return files


# ---------------------------
# download_dataframe_from_src
# ---------------------------
def download_dataframe_from_src(
    src: str, tmp_dir: Path, *, retries: int = 2
) -> Tuple[Any, Optional[int]]:
    pd = ensure_pandas()

    # chemprop_zenodo/* handling (stream or persistent cache)
    if str(src).startswith("chemprop_zenodo/"):
        persistent_cache = os.environ.get("CHEMPROP_EXTRACT_DIR")
        if persistent_cache:
            extract_root = Path(persistent_cache)
            remove_after = False
        else:
            extract_root = tmp_dir.joinpath("chemprop_extracted")
            remove_after = True

        desired_subpath = Path(src.replace("chemprop_zenodo/", ""))
        target_dir = extract_root.joinpath(*desired_subpath.parts)

        # quick check if already present
        if target_dir.exists():
            csvs = list(target_dir.rglob("*.csv"))
            if csvs:
                dfs = []
                for p in csvs:
                    try:
                        dfs.append(pd.read_csv(p, low_memory=False))
                    except Exception:
                        try:
                            dfs.append(
                                pd.read_csv(p, compression="infer", low_memory=False)
                            )
                        except Exception:
                            logger.warning("Could not parse CSV %s", p)
                if dfs:
                    if remove_after:
                        try:
                            shutil.rmtree(extract_root)
                        except Exception:
                            pass
                    return pd.concat(dfs, ignore_index=True), None

        # stream the bundle and extract targeted tokens
        bundle_url = DEFAULT_CONFIG.get("chemprop_bundle", {}).get("src")
        if not bundle_url:
            raise RuntimeError(
                "chemprop_zenodo path requested but chemprop_bundle URL missing in DEFAULT_CONFIG"
            )

        # infer tokens while avoiding generic tokens like 'data'
        parts = [
            p
            for p in desired_subpath.parts
            if p and p.lower() not in ("", ".", "/", "data", "chemprop")
        ]
        want_tokens: List[str] = [p for p in parts if "barriers" in p.lower()]
        if not want_tokens and parts:
            want_tokens = [parts[-1]]
        if not want_tokens:
            want_tokens = [
                "barriers_e2",
                "barriers_sn2",
                "barriers_cycloadd",
                "barriers_rdb7",
                "barriers_rgd1",
            ]
        want_tokens = [str(t) for t in want_tokens]

        logger.info("[chemprop] streaming bundle for tokens: %s", want_tokens)
        try:
            stream_extract_selected_from_targz_safe(
                bundle_url,
                want_tokens,
                dest_dir=str(extract_root),
                timeout=600,
                max_no_progress=200000,
                min_matches_per_target=1,
                verbose=True,
            )
        except Exception as e:
            logger.warning("chemprop streaming extraction raised: %s", e)

        # search for candidate directories containing tokens, score by CSV count
        candidate_dirs: List[Path] = []
        for token in want_tokens:
            found = list(extract_root.rglob(f"*{token}*"))
            found_dirs = [p for p in found if p.is_dir()]
            candidate_dirs.extend(found_dirs)

        unique_candidates: Dict[Path, int] = {}
        for d in candidate_dirs:
            if d in unique_candidates:
                continue
            csv_count = sum(1 for _ in d.rglob("*.csv"))
            if csv_count:
                unique_candidates[d] = csv_count

        if unique_candidates:
            best_dir = max(unique_candidates.items(), key=lambda kv: kv[1])[0]
            csvs = list(best_dir.rglob("*.csv"))
            if csvs:
                dfs = []
                for p in csvs:
                    try:
                        dfs.append(pd.read_csv(p, low_memory=False))
                    except Exception:
                        try:
                            dfs.append(
                                pd.read_csv(p, compression="infer", low_memory=False)
                            )
                        except Exception:
                            logger.warning("Could not parse CSV %s", p)
                if dfs:
                    if remove_after:
                        try:
                            shutil.rmtree(extract_root)
                        except Exception:
                            pass
                    return pd.concat(dfs, ignore_index=True), None

        # fallback: any CSV under extract_root
        any_csvs = list(extract_root.rglob("*.csv"))
        if any_csvs:
            dfs = []
            for p in any_csvs:
                try:
                    dfs.append(pd.read_csv(p, low_memory=False))
                except Exception:
                    try:
                        dfs.append(
                            pd.read_csv(p, compression="infer", low_memory=False)
                        )
                    except Exception:
                        logger.warning("Could not parse CSV %s", p)
            if dfs:
                if remove_after:
                    try:
                        shutil.rmtree(extract_root)
                    except Exception:
                        pass
                return pd.concat(dfs, ignore_index=True), None

        # nothing found -> cleanup if temp and fail
        if remove_after:
            try:
                shutil.rmtree(extract_root)
            except Exception:
                pass
        raise RuntimeError(
            f"Expected extracted chemprop data under {target_dir}, but it's missing after streaming extraction"
        )

    # raw GitHub directory listing
    if ("raw.githubusercontent.com" in src) or (
        src.startswith("https://github.com/") and "/raw/" in src
    ):
        files = list_github_dir(src)
        if files:
            dfs = []
            for furl in files:
                try:
                    content, status = download_to_bytes(furl)
                    try:
                        df = read_bytes_as_dataframe(content, src_hint=furl)
                        dfs.append(df)
                    except Exception as e:
                        logger.warning("Skipping file %s: %s", furl, e)
                except Exception as e:
                    logger.warning("Failed downloading %s: %s", furl, e)
            if dfs:
                return pd.concat(dfs, ignore_index=True), None

    # http(s) direct handling (ZIP/TAR/CSV/JSON/XLSX)
    if src.startswith("http://") or src.startswith("https://"):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                content, status = download_to_bytes(src)
                lower = src.lower()
                # try ZIP
                if lower.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
                    try:
                        df = parse_zip_bytes_to_dataframe(
                            content, target_basename="SNAR_reaction_dataset_SI.csv"
                        )
                        return df, status
                    except Exception as e_zip:
                        logger.warning(
                            "ZIP parse failed: %s — trying temp extract", e_zip
                        )
                        tmp_zip_dir = tmp_dir.joinpath("zip_extract_tmp")
                        if tmp_zip_dir.exists():
                            shutil.rmtree(tmp_zip_dir)
                        tmp_zip_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            z = zipfile.ZipFile(io.BytesIO(content))
                            dfs = []
                            for m in z.namelist():
                                if m.endswith("/"):
                                    continue
                                outp = tmp_zip_dir.joinpath(m)
                                outp.parent.mkdir(parents=True, exist_ok=True)
                                with z.open(m) as s, open(outp, "wb") as d:
                                    shutil.copyfileobj(s, d)
                                try:
                                    if outp.suffix.lower() in (".csv", ".gz"):
                                        dfs.append(
                                            ensure_pandas().read_csv(
                                                outp,
                                                compression="infer",
                                                low_memory=False,
                                            )
                                        )
                                    elif outp.suffix.lower() in (".xlsx", ".xls"):
                                        dfs.append(ensure_pandas().read_excel(outp))
                                    elif outp.suffix.lower() == ".json":
                                        dfs.append(ensure_pandas().read_json(outp))
                                except Exception as e_p:
                                    logger.warning("Failed to parse %s: %s", outp, e_p)
                            if dfs:
                                return (
                                    ensure_pandas().concat(dfs, ignore_index=True),
                                    status,
                                )
                        finally:
                            if tmp_zip_dir.exists():
                                shutil.rmtree(tmp_zip_dir)
                # try tar.gz
                if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
                    try:
                        with tarfile.open(
                            fileobj=io.BytesIO(content), mode="r:gz"
                        ) as tar:
                            for member in tar.getmembers():
                                if not member.isfile():
                                    continue
                                name = member.name
                                if any(
                                    name.lower().endswith(ext)
                                    for ext in (
                                        ".csv",
                                        ".csv.gz",
                                        ".json",
                                        ".xlsx",
                                        ".xls",
                                    )
                                ):
                                    f = tar.extractfile(member)
                                    if f:
                                        b = f.read()
                                        try:
                                            df = read_bytes_as_dataframe(
                                                b, src_hint=name
                                            )
                                            return df, status
                                        except Exception:
                                            pass
                    except Exception as e:
                        logger.warning("tar.gz parse failed for %s: %s", src, e)
                # try plain bytes -> df
                try:
                    df = read_bytes_as_dataframe(content, src_hint=src)
                    return df, status
                except Exception as e_parse:
                    try:
                        text = content.decode("utf-8", errors="replace")
                        df = ensure_pandas().read_csv(
                            io.StringIO(text), low_memory=False
                        )
                        return df, status
                    except Exception:
                        raise e_parse
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d failed to download/parse %s: %s", attempt, src, exc
                )
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"Failed to download {src}: {last_exc}")

    # local path handling
    p = Path(src)
    if p.exists():
        if p.is_dir():
            pd = ensure_pandas()
            dfs = []
            for suf in ("*.csv", "*.csv.gz", "*.json", "*.xlsx", "*.xls"):
                for f in p.rglob(suf):
                    try:
                        if f.suffix.lower() in (".csv", ".gz"):
                            dfs.append(
                                pd.read_csv(f, compression="infer", low_memory=False)
                            )
                        elif f.suffix.lower() in (".xlsx", ".xls"):
                            dfs.append(pd.read_excel(f))
                        else:
                            dfs.append(pd.read_json(f))
                    except Exception as e:
                        logger.warning("Failed to read %s: %s", f, e)
            if dfs:
                return pd.concat(dfs, ignore_index=True), None
            raise RuntimeError(f"No readable files under directory {p}")
        else:
            pd = ensure_pandas()
            if p.suffix.lower() in (".csv", ".gz"):
                return pd.read_csv(p, compression="infer", low_memory=False), None
            if p.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(p), None
            if p.suffix.lower() == ".json":
                return pd.read_json(p), None
            text = p.read_text(encoding="utf-8")
            try:
                return pd.read_csv(io.StringIO(text), low_memory=False), None
            except Exception:
                try:
                    return pd.read_json(io.StringIO(text)), None
                except Exception:
                    raise RuntimeError(f"Cannot parse local file {p}")
    else:
        raise RuntimeError(f"Local path not found: {src}")


# ---------------------------
# Per-entry processing
# ---------------------------
def process_property_entry(
    cfg: Dict[str, Dict[str, str]],
    name: str,
    tmp_dir: Path,
    *,
    dry_run: bool = False,
    retries: int = 2,
) -> Dict[str, Any]:
    pd = ensure_pandas()
    entry = cfg[name]
    src = entry["src"]
    out = entry.get("out") or f"Data/property/{name}.csv.gz"
    start = time.time()
    result: Dict[str, Any] = {
        "name": name,
        "src": src,
        "out": out,
        "status": "failed",
        "message": "",
        "input_rows": None,
        "output_rows": None,
        "saved": False,
        "time_s": None,
        "columns": json.dumps([], ensure_ascii=False),
        "columns_preview": "",
        "columns_full": "",
    }

    try:
        # Skip heavy chemprop bundle in dry-run
        if dry_run and name == "chemprop_bundle":
            result.update(
                {
                    "status": "skipped",
                    "message": "dry-run: skipped heavy bundle download/extract",
                    "time_s": 0.0,
                }
            )
            return result

        # COMBINE(...) support
        if isinstance(src, str) and src.startswith("COMBINE(") and src.endswith(")"):
            inner = src[len("COMBINE(") : -1]
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            dfs = []
            for pkey in parts:
                if pkey not in cfg:
                    raise RuntimeError(
                        f"Referenced combine entry '{pkey}' not found in config"
                    )
                ref_out = cfg[pkey].get("out")
                if ref_out and Path(ref_out).exists():
                    dfs.append(
                        pd.read_csv(ref_out, compression="infer", low_memory=False)
                    )
                else:
                    ref_df, _ = download_dataframe_from_src(
                        cfg[pkey]["src"], tmp_dir, retries=retries
                    )
                    if isinstance(ref_df, list):
                        raise RuntimeError(
                            f"Referenced entry '{pkey}' did not produce a DataFrame"
                        )
                    dfs.append(ref_df)
            if not dfs:
                raise RuntimeError("COMBINE resulted in no data")
            df = pd.concat(dfs, ignore_index=True)
        else:
            df_or_obj, _ = download_dataframe_from_src(src, tmp_dir, retries=retries)
            if isinstance(df_or_obj, list) and all(
                isinstance(x, Path) for x in df_or_obj
            ):
                dfs = []
                for p in df_or_obj:
                    try:
                        if p.suffix.lower() in (".csv", ".gz"):
                            dfs.append(
                                pd.read_csv(p, compression="infer", low_memory=False)
                            )
                    except Exception:
                        pass
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    raise RuntimeError("Downloaded archive but no CSVs could be parsed")
            elif isinstance(df_or_obj, bytes):
                df = read_bytes_as_dataframe(df_or_obj, src_hint=src)
            else:
                df = df_or_obj

        if df is None:
            raise RuntimeError("No dataframe produced")

        result["input_rows"] = int(getattr(df, "shape", (None, None))[0])
        result["output_rows"] = int(getattr(df, "shape", (None, None))[0])

        cols = list(df.columns)
        result["columns"] = json.dumps(cols, ensure_ascii=False)
        result["columns_preview"] = ", ".join(cols[:10])
        result["columns_full"] = ";".join(cols)

        if not dry_run:
            save_df_gz(df, out)
            result["saved"] = True
        else:
            result["saved"] = False

        result["status"] = "success"
        result["message"] = f"OK; produced {result['output_rows']} rows"

    except Exception as exc:
        logger.exception("Entry %s failed: %s", name, exc)
        msg = str(exc)
        if len(msg) > 400:
            msg = msg[:400] + "...(truncated)"
        result["message"] = msg
    finally:
        result["time_s"] = round(time.time() - start, 3)
    return result


# ---------------------------
# Summary display
# ---------------------------
def _print_summary_table(results: List[Dict[str, Any]]):
    if not results:
        print("No entries processed — nothing to display.")
        return
    try:
        from tabulate import tabulate  # type: ignore

        cols = [
            "name",
            "status",
            "input_rows",
            "output_rows",
            "columns_preview",
            "columns_full",
            "saved",
            "time_s",
            "message",
        ]
        table = [[r.get(c) for c in cols] for r in results]
        print(tabulate(table, headers=cols, tablefmt="github"))
    except Exception:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(results)
        desired = [
            "name",
            "status",
            "input_rows",
            "output_rows",
            "columns_preview",
            "columns_full",
            "saved",
            "time_s",
            "message",
        ]
        display_cols = [c for c in desired if c in df.columns]
        if display_cols:
            print(df[display_cols].to_string(index=False))
        else:
            print("Summary (full):")
            print(df.to_string(index=False))


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Build property datasets (curate -> save) with summary."
    )
    p.add_argument(
        "--config", help="JSON/YAML config file; if omitted DEFAULT_CONFIG is used."
    )
    p.add_argument(
        "--entries",
        help="Comma-separated subset of config keys to process (default: all).",
    )
    p.add_argument(
        "--src", help="Process a single source URL/local_path (overrides config)."
    )
    p.add_argument("--out", help="Output path when using --src (required).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except saving files. chemprop_bundle is skipped.",
    )
    p.add_argument("--retries", type=int, default=2, help="Load/process retries.")
    p.add_argument("--write-default", help="Write DEFAULT_CONFIG to file and exit.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    p.add_argument(
        "--summary-out",
        default="reports/property_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_property_dataset starting")

    if args.write_default:
        outp = Path(args.write_default)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        logger.info("Wrote default config to %s", outp)
        return

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config file not found: {cfg_path}")
        text = cfg_path.read_text(encoding="utf-8")
        try:
            cfg = json.loads(text)
        except Exception:
            import yaml  # type: ignore

            cfg = yaml.safe_load(text)
    else:
        cfg = DEFAULT_CONFIG

    entries: Optional[List[str]] = None
    if args.entries:
        requested = [e.strip() for e in args.entries.split(",") if e.strip()]
        missing = [e for e in requested if e not in cfg]
        available = [e for e in requested if e in cfg]
        if missing:
            logger.warning(
                "Requested entries not found in config and will be skipped: %s", missing
            )
        if not available:
            raise SystemExit(f"No matching entries for --entries: {requested}")
        entries = available
        logger.info("Processing subset entries (validated): %s", entries)

    tmp_dir = Path(".build_property_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # process chemprop_bundle first if present
    keys = list(cfg.keys())
    if "chemprop_bundle" in keys:
        keys = ["chemprop_bundle"] + [k for k in keys if k != "chemprop_bundle"]

    for name in keys:
        if entries and name not in entries:
            logger.debug("Skipping %s (not requested)", name)
            continue
        if name not in cfg:
            logger.debug("Skipping %s (not in config)", name)
            continue
        try:
            res = process_property_entry(
                cfg, name, tmp_dir, dry_run=args.dry_run, retries=args.retries
            )
        except Exception as exc:
            logger.exception("Failed to process %s: %s", name, exc)
            res = {
                "name": name,
                "src": cfg[name].get("src"),
                "out": cfg[name].get("out"),
                "status": "failed",
                "message": str(exc)[:400],
                "input_rows": None,
                "output_rows": None,
                "saved": False,
                "time_s": None,
                "columns": json.dumps([], ensure_ascii=False),
                "columns_preview": "",
                "columns_full": "",
            }
        results.append(res)

    _print_summary_table(results)
    pd = ensure_pandas()
    summary_df = pd.DataFrame(results)
    outp = Path(args.summary_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outp, index=False, compression="gzip")
    logger.info("Wrote summary to %s", outp)

    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    except Exception:
        pass

    logger.info("build_property_dataset finished")


if __name__ == "__main__":
    main()
