"""
synrxn.data_loader
------------------

DataLoader: Zenodo-first dataset loader with optional GitHub fallback.

Immutable defaults:
- Concept DOI: 10.5281/zenodo.17297258
- GitHub owner/repo: TieuLongPhan / SynRXN

Usage examples are in the class docstring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote as urlquote
import io
import re
import math
import hashlib
import requests
import pandas as pd
import zipfile
import tarfile

# -----------------------
# Immutable module defaults
# -----------------------
CONCEPT_DOI = "10.5281/zenodo.17297258"
GH_OWNER = "TieuLongPhan"
GH_REPO = "SynRXN"

_ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"
_ZENODO_SEARCH_API = "https://zenodo.org/api/records"
_GH_RAW_TPL = (
    "https://raw.githubusercontent.com/{owner}/{repo}/refs/{ref_type}/{ref}/Data"
)
_GH_API_TPL = (
    "https://api.github.com/repos/{owner}/{repo}/contents/Data/{task}?ref={ref}"
)


class DataLoader:
    """
    Object-oriented loader for CSV(.gz) datasets stored in the SynRXN Data/ tree.
    Zenodo-first (pinned by version via concept DOI), optional GitHub raw fallback.

    :param task: Subfolder under `Data/` (e.g. "aam", "rbl", "class", "prop", "synthesis").
    :param version: Target version label to pin (e.g. "0.0.5" or "v0.0.5").
                    If ``None``, resolves to the latest published version under the concept DOI.
    :param cache_dir: Optional local cache directory. If provided, gz payloads are cached as
                      ``{cache_dir}/{task}__{name}.csv.gz``.
    :param timeout: HTTP request timeout in seconds (default: 20).
    :param user_agent: HTTP User-Agent header used for requests (default: "SynRXN-DataLoader/1.4").
    :param max_workers: Maximum number of worker threads used by :meth:`load_many`.
    :param gh_ref: Optional explicit GitHub ref (branch/tag/commit) to use for fallback. If omitted
                   and `version` is provided, loader will try tags ``v{version}`` then ``{version}``
                   and finally ``main``. If omitted and `version` is ``None``, loader tries ``main``.
    :param gh_enable: If False, disables any GitHub network calls (useful for restricted environments).
    :param fallback: If True, attempt GitHub fallback when the file isn't present (or extractable)
                     from the Zenodo record. If False, only Zenodo is used and the loader raises a
                     FileNotFoundError if the file is not present in the pinned Zenodo record.

    :raises RuntimeError: If the concept DOI has no matching Zenodo records, or if the requested
                          `version` label cannot be found under the concept DOI.

    Examples
    --------
    Pin to a specific Zenodo version, no GitHub fallback:

    >>> dl = DataLoader(task="aam", version="0.0.5", cache_dir=Path("/tmp/cache"), fallback=False)
    >>> df = dl.load("ecoli")

    Use latest Zenodo version, allow GitHub fallback to `main`:

    >>> dl = DataLoader(task="aam", version=None, cache_dir=Path("/tmp/cache"), fallback=True)
    >>> dl.print_zenodo_files(limit=200)
    >>> df = dl.load("some_name")

    Parallel-loading multiple datasets:

    >>> dl = DataLoader(task="aam", version="0.0.5", fallback=True)
    >>> dfs = dl.load_many(["ecoli", "uspto_50k"], parallel=True)
    """

    def __init__(
        self,
        task: str,
        version: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: int = 20,
        user_agent: str = "SynRXN-DataLoader/1.4",
        max_workers: int = 6,
        gh_ref: Optional[str] = None,
        gh_enable: bool = True,
        fallback: bool = True,
    ) -> None:
        self.task = str(task).strip("/")
        self.version = version.strip() if isinstance(version, str) else None
        self.timeout = int(timeout)
        self.headers = {"User-Agent": user_agent}
        self.max_workers = int(max_workers)

        # GitHub/fallback configuration (owner/repo are module-level constants and immutable)
        self.gh_enable = bool(gh_enable)
        self.gh_owner = GH_OWNER
        self.gh_repo = GH_REPO
        self.fallback = bool(fallback)

        # Build ordered list of GitHub refs to try for fallback
        self._gh_try_refs: List[Tuple[str, str]] = []
        if self.gh_enable:
            if gh_ref:
                self._gh_try_refs = [("heads", gh_ref)]
            elif self.version:
                self._gh_try_refs = [
                    ("tags", f"v{self.version}"),
                    ("tags", self.version),
                    ("heads", "main"),
                ]
            else:
                self._gh_try_refs = [("heads", "main")]

        # Cache path
        self.cache_dir: Optional[Path] = (
            Path(cache_dir).expanduser().resolve() if cache_dir else None
        )
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve Zenodo record id and build file index (may raise if DOI not found)
        self._record_id: Optional[int] = self._resolve_record_id(
            CONCEPT_DOI, self.version
        )
        self._file_index: Dict[str, Dict] = self._build_file_index(self._record_id)

        # name caches
        self._names_cache_zenodo: Optional[List[str]] = None
        self._names_cache_github: Optional[List[str]] = None

    # ---------------------------
    # Introspection & listing
    # ---------------------------
    def __repr__(self) -> str:
        return (
            f"DataLoader(task={self.task!r}, version={self.version!r}, record={self._record_id}, "
            f"fallback={self.fallback}, gh_refs={self._gh_try_refs}, cache_dir={self.cache_dir})"
        )

    def print_zenodo_files(self, limit: int = 200) -> None:
        """Print file keys available in the resolved Zenodo record (helpful for debugging)."""
        keys = list(self._file_index.keys())
        print(f"Zenodo record {self._record_id} has {len(keys)} files.")
        for k in keys[:limit]:
            print(" ", k)
        if len(keys) > limit:
            print("  ... (remaining files elided)")

    def find_zenodo_keys(self, term: str) -> List[str]:
        """Return Zenodo file keys that contain `term` (case-insensitive)."""
        t = term.lower()
        return [k for k in self._file_index.keys() if t in k.lower()]

    @property
    def names(self) -> List[str]:
        """Combined list of available dataset base names (Zenodo + optional GitHub fallback)."""
        return self.available_names()

    def available_names(self, refresh: bool = False) -> List[str]:
        """Return sorted combined names from Zenodo and GitHub (if fallback enabled)."""
        z_names = self._available_names_zenodo(refresh=refresh)
        g_names = (
            self._available_names_github(refresh=refresh)
            if (self.gh_enable and self.fallback)
            else []
        )
        return sorted(set(z_names).union(g_names))

    def refresh_names(self) -> List[str]:
        """Force re-fetch of Zenodo record index and GitHub listing caches."""
        self._file_index = self._build_file_index(self._record_id)
        self._names_cache_zenodo = None
        self._names_cache_github = None
        return self.available_names(refresh=True)

    def suggest(self, name: str, n: int = 5) -> List[str]:
        """Return close name matches from combined available names."""
        import difflib

        names = self.available_names()
        if not names:
            return []
        return difflib.get_close_matches(name, names, n=n, cutoff=0.4)

    def print_names(self, cols: int = 3, show_count: bool = True) -> None:
        """Pretty print available names in columns."""
        names = self.available_names()
        if show_count:
            print(f"Datasets in task '{self.task}': {len(names)}")
        if not names:
            print("  (no names found)")
            return
        rows = math.ceil(len(names) / cols)
        padded = names + [""] * (rows * cols - len(names))
        matrix = [padded[i : i + rows] for i in range(0, rows * cols, rows)]
        for r in range(rows):
            row_items = [matrix[c][r].ljust(30) for c in range(cols) if matrix[c][r]]
            print("  " + "  ".join(row_items))

    # ---------------------------
    # Core loading
    # ---------------------------
    def load(
        self,
        name: str,
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        **pd_kw,
    ) -> pd.DataFrame:
        """
        Load Data/<task>/<name>.csv(.gz) from Zenodo (preferred) or GitHub raw fallback.

        The function will:
         - look for exact Data/<task>/<name>.csv.gz then .csv in the pinned Zenodo record
         - try fuzzy CSV filename matches in the record
         - try extracting the dataset from archives (zip / tar) attached to the Zenodo record
         - if allowed (fallback=True) try GitHub raw under configured refs
         - verify checksums when present and cache gz bytes if cache_dir given

        Raises FileNotFoundError with helpful diagnostics if no source contains the dataset.
        """
        rel_gz = f"Data/{self.task}/{name}.csv.gz"
        rel_csv = f"Data/{self.task}/{name}.csv"
        tried: List[str] = []
        last_err = None

        def _read_buf(content: bytes, ext: str) -> pd.DataFrame:
            buf = io.BytesIO(content)
            if ext == ".csv.gz":
                return pd.read_csv(buf, compression="gzip", dtype=dtype, **pd_kw)
            else:
                return pd.read_csv(buf, compression=None, dtype=dtype, **pd_kw)

        def _get_download_link_from_meta(meta: Dict) -> Optional[str]:
            """Return a usable download link from a Zenodo file metadata entry or construct one."""
            links = meta.get("links", {}) or {}
            dl = links.get("download")
            if dl:
                return dl
            # try 'self' or 'html' (may be usable)
            for alt in ("self", "html"):
                if alt in links:
                    return links.get(alt)
            # construct common Zenodo file URL when only 'key' present
            key = meta.get("key", "")
            if key and self._record_id is not None:
                filename = key.split("/")[-1]
                return f"https://zenodo.org/record/{self._record_id}/files/{urlquote(filename)}?download=1"
            return None

        # If file index is empty, attempt one re-build (network glitch or unusual record shape)
        if not self._file_index:
            try:
                self._file_index = self._build_file_index(self._record_id)
            except Exception as e:
                last_err = e

        # If still empty and fallback is allowed, try GitHub raw directly (skip Zenodo attempts)
        if not self._file_index:
            if self.fallback and self.gh_enable:
                for ref_type, ref in self._gh_try_refs:
                    base = _GH_RAW_TPL.format(
                        owner=self.gh_owner,
                        repo=self.gh_repo,
                        ref_type=ref_type,
                        ref=ref,
                    )
                    for ext in (".csv.gz", ".csv"):
                        url = f"{base}/{self.task}/{name}{ext}"
                        tried.append(url)
                        try:
                            resp = requests.get(
                                url, headers=self.headers, timeout=self.timeout
                            )
                            if resp.status_code == 200:
                                content = resp.content
                                if ext == ".csv.gz" and use_cache and self.cache_dir:
                                    try:
                                        (
                                            self.cache_dir
                                            / f"{self.task}__{name}.csv.gz"
                                        ).write_bytes(content)
                                    except Exception:
                                        pass
                                return _read_buf(content, ext)
                            else:
                                last_err = RuntimeError(
                                    f"HTTP {resp.status_code} for {url}"
                                )
                        except Exception as e:
                            last_err = e
            # No Zenodo files and GitHub fallback failed or disabled -> user-visible guidance
            msg = [
                f"Failed to fetch dataset '{name}' for task '{self.task}'.",
                f"Concept DOI: {CONCEPT_DOI}",
                f"Version: {self.version or 'latest'} (record {self._record_id})",
                "",
                "Zenodo record contained no listed files (empty 'files' array).",
                "If this is unexpected, run `dl.print_zenodo_files()` to inspect the record,",
                "or set fallback=True to allow fetching directly from GitHub raw tree.",
            ]
            if last_err:
                msg += ["", f"Last error: {last_err!s}"]
            raise FileNotFoundError("\n".join(msg))

        # 1) exact match (prefer .csv.gz)
        if rel_gz in self._file_index or rel_csv in self._file_index:
            key = rel_gz if rel_gz in self._file_index else rel_csv
            meta = self._file_index[key]
            dl_link = _get_download_link_from_meta(meta)
            if not dl_link:
                tried.append(f"(no usable download link for Zenodo file entry: {key})")
                last_err = RuntimeError(
                    f"Missing download link for Zenodo file key {key}"
                )
            else:
                tried.append(dl_link)
                try:
                    resp = requests.get(
                        dl_link, headers=self.headers, timeout=self.timeout
                    )
                    resp.raise_for_status()
                    content = resp.content
                    algo, hexdigest = self._parse_checksum(meta.get("checksum", ""))
                    if (
                        algo
                        and hexdigest
                        and not self._verify_checksum(content, algo, hexdigest)
                    ):
                        raise RuntimeError("Zenodo checksum mismatch")
                    ext = ".csv.gz" if key.endswith(".csv.gz") else ".csv"
                    if use_cache and self.cache_dir and ext == ".csv.gz":
                        try:
                            (
                                self.cache_dir / f"{self.task}__{name}.csv.gz"
                            ).write_bytes(content)
                        except Exception:
                            pass
                    return _read_buf(content, ext)
                except Exception as e:
                    last_err = e

        # 2) fuzzy CSV members in the Zenodo record
        candidates = (
            self.find_zenodo_keys(f"{self.task}/{name}")
            + self.find_zenodo_keys(f"{self.task}_{name}")
            + self.find_zenodo_keys(name)
        )
        seen: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        candidates = seen
        for key in candidates:
            if key.endswith(".csv") or key.endswith(".csv.gz"):
                meta = self._file_index[key]
                dl_link = _get_download_link_from_meta(meta)
                if not dl_link:
                    tried.append(
                        f"(no usable download link for Zenodo file entry: {key})"
                    )
                    last_err = RuntimeError(
                        f"Missing download link for Zenodo file key {key}"
                    )
                    continue
                tried.append(dl_link)
                try:
                    resp = requests.get(
                        dl_link, headers=self.headers, timeout=self.timeout
                    )
                    resp.raise_for_status()
                    content = resp.content
                    algo, hexdigest = self._parse_checksum(meta.get("checksum", ""))
                    if (
                        algo
                        and hexdigest
                        and not self._verify_checksum(content, algo, hexdigest)
                    ):
                        raise RuntimeError("Zenodo checksum mismatch")
                    ext = ".csv.gz" if key.endswith(".csv.gz") else ".csv"
                    if use_cache and self.cache_dir and ext == ".csv.gz":
                        try:
                            (
                                self.cache_dir / f"{self.task}__{name}.csv.gz"
                            ).write_bytes(content)
                        except Exception:
                            pass
                    return _read_buf(content, ext)
                except Exception as e:
                    last_err = e
                    continue

        # 3) archives attached to record: try to extract Data/<task>/<name>.csv(.gz)
        archive_keys = [
            k
            for k in self._file_index.keys()
            if k.lower().endswith((".zip", ".tar.gz", ".tgz", ".tar"))
        ]
        for ak in archive_keys:
            meta = self._file_index[ak]
            dl_link = _get_download_link_from_meta(meta)
            if not dl_link:
                tried.append(
                    f"(no usable download link for Zenodo archive entry: {ak})"
                )
                last_err = RuntimeError(
                    f"Missing download link for Zenodo archive key {ak}"
                )
                continue
            tried.append(dl_link)
            try:
                r = requests.get(dl_link, headers=self.headers, timeout=self.timeout)
                r.raise_for_status()
                arch_bytes = r.content
                algo, hexdigest = self._parse_checksum(meta.get("checksum", ""))
                if (
                    algo
                    and hexdigest
                    and not self._verify_checksum(arch_bytes, algo, hexdigest)
                ):
                    raise RuntimeError("Zenodo archive checksum mismatch")
                # zip
                if ak.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(arch_bytes)) as z:
                        exact_candidates = [
                            f"Data/{self.task}/{name}.csv.gz",
                            f"Data/{self.task}/{name}.csv",
                        ]
                        members = z.namelist()
                        member = None
                        for ec in exact_candidates:
                            if ec in members:
                                member = ec
                                break
                        if member is None:
                            members_lower = [m.lower() for m in members]
                            for idx, m in enumerate(members_lower):
                                if (
                                    f"{self.task}/{name}".lower() in m
                                    or f"{self.task}_{name}".lower() in m
                                    or name.lower() in m
                                ):
                                    member = members[idx]
                                    break
                        if member:
                            with z.open(member) as mf:
                                content = mf.read()
                                if member.endswith(".csv.gz"):
                                    return pd.read_csv(
                                        io.BytesIO(content),
                                        compression="gzip",
                                        dtype=dtype,
                                        **pd_kw,
                                    )
                                else:
                                    return pd.read_csv(
                                        io.BytesIO(content),
                                        compression=None,
                                        dtype=dtype,
                                        **pd_kw,
                                    )
                else:
                    # tar / tgz
                    with tarfile.open(
                        fileobj=io.BytesIO(arch_bytes), mode="r:*"
                    ) as tar:
                        member = None
                        exact_candidates = [
                            f"Data/{self.task}/{name}.csv.gz",
                            f"Data/{self.task}/{name}.csv",
                        ]
                        members = tar.getmembers()
                        names_list = [m.name for m in members]
                        for ec in exact_candidates:
                            if ec in names_list:
                                member = ec
                                break
                        if member is None:
                            for m in names_list:
                                ml = m.lower()
                                if (
                                    f"{self.task}/{name}".lower() in ml
                                    or f"{self.task}_{name}".lower() in ml
                                    or name.lower() in ml
                                ):
                                    member = m
                                    break
                        if member:
                            fobj = tar.extractfile(member)
                            if fobj is None:
                                continue
                            content = fobj.read()
                            if member.endswith(".csv.gz"):
                                return pd.read_csv(
                                    io.BytesIO(content),
                                    compression="gzip",
                                    dtype=dtype,
                                    **pd_kw,
                                )
                            else:
                                return pd.read_csv(
                                    io.BytesIO(content),
                                    compression=None,
                                    dtype=dtype,
                                    **pd_kw,
                                )
            except Exception as e:
                last_err = e
                continue

        # 4) GitHub raw fallback (if enabled)
        if self.fallback and self.gh_enable:
            for ref_type, ref in self._gh_try_refs:
                base = _GH_RAW_TPL.format(
                    owner=self.gh_owner, repo=self.gh_repo, ref_type=ref_type, ref=ref
                )
                for ext in (".csv.gz", ".csv"):
                    url = f"{base}/{self.task}/{name}{ext}"
                    tried.append(url)
                    try:
                        resp = requests.get(
                            url, headers=self.headers, timeout=self.timeout
                        )
                        if resp.status_code == 200:
                            content = resp.content
                            if ext == ".csv.gz" and use_cache and self.cache_dir:
                                try:
                                    (
                                        self.cache_dir / f"{self.task}__{name}.csv.gz"
                                    ).write_bytes(content)
                                except Exception:
                                    pass
                            return _read_buf(content, ext)
                        else:
                            last_err = RuntimeError(
                                f"HTTP {resp.status_code} for {url}"
                            )
                    except Exception as e:
                        last_err = e

        # not found -> raise with diagnostics
        avail = self.available_names(refresh=True)
        suggestions = self.suggest(name) if avail else []
        msg_lines: List[str] = [
            f"Failed to fetch dataset '{name}' for task '{self.task}'.",
            f"Concept DOI: {CONCEPT_DOI}",
            f"Version: {self.version or 'latest'} (record {self._record_id})",
            "Tried URLs / archives:",
        ]
        if tried:
            msg_lines += [f"  {u}" for u in tried]
        else:
            msg_lines += ["  (no candidate URLs/archives found)"]

        if avail:
            msg_lines.append("")
            msg_lines.append(
                "Available dataset names (from Zenodo/GitHub where applicable):"
            )
            display = avail[:200] if len(avail) > 200 else avail
            msg_lines += [f"  {n}" for n in display]
            if suggestions:
                msg_lines.append("")
                msg_lines.append(f"Did you mean: {suggestions} ?")

        if last_err:
            msg_lines.append("")
            msg_lines.append(f"Last error: {last_err!s}")

        raise FileNotFoundError("\n".join(msg_lines))

    # ---------------------------
    # Parallel loading helper
    # ---------------------------
    def load_many(
        self,
        names: Iterable[str],
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        parallel: bool = True,
        **pd_kw,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets and return a dict name -> DataFrame.

        If parallel=True (default) uses ThreadPoolExecutor up to max_workers.
        """
        names_list = list(names)
        results: Dict[str, pd.DataFrame] = {}

        if not parallel or self.max_workers <= 1 or len(names_list) == 1:
            for nm in names_list:
                try:
                    results[nm] = self.load(
                        nm, use_cache=use_cache, dtype=dtype, **pd_kw
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load {self.task}/{nm}: {e}") from e
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(self.load, nm, use_cache, dtype, **pd_kw): nm
                for nm in names_list
            }
            for fut in as_completed(futures):
                nm = futures[fut]
                try:
                    results[nm] = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed to load {self.task}/{nm}: {e}") from e
        return results

    # ---------------------------
    # Zenodo index & utilities
    # ---------------------------
    def _resolve_record_id(self, concept_doi: str, version: Optional[str]) -> int:
        """
        Resolve concept DOI -> specific record id (pinned version). If version is None,
        return the most recently updated version under the concept DOI.
        """
        params = {"q": f'conceptdoi:"{concept_doi}"', "all_versions": 1, "size": 200}
        r = requests.get(
            _ZENODO_SEARCH_API,
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            raise RuntimeError(f"No Zenodo records found for concept DOI {concept_doi}")
        if version:
            target = self._normalize_version(version)
            for h in hits:
                meta_ver = self._normalize_version(
                    h.get("metadata", {}).get("version", "")
                )
                if meta_ver == target:
                    return int(h["id"])
            # fallback raw compare
            for h in hits:
                raw = str(h.get("metadata", {}).get("version", "")).strip()
                if raw == version or raw == f"v{version}" or f"v{raw}" == version:
                    return int(h["id"])
            raise RuntimeError(
                f"Version '{version}' not found under {concept_doi}. "
                f"Available: {sorted({h.get('metadata', {}).get('version','') for h in hits})}"
            )
        # return most recently updated
        hits_sorted = sorted(
            hits, key=lambda h: h.get("updated", h.get("created", "")), reverse=True
        )
        return int(hits_sorted[0]["id"])

    def _build_file_index(self, record_id: Optional[int]) -> Dict[str, Dict]:
        """Return mapping key -> file metadata for given Zenodo record id."""
        if record_id is None:
            return {}
        url = _ZENODO_RECORD_API.format(record_id=record_id)
        r = requests.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        meta = r.json()
        files = meta.get("files", [])
        # index by 'key' (Zenodo's filename path)
        return {f.get("key", ""): f for f in files if f.get("key")}

    def _parse_checksum(
        self, checksum_field: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse Zenodo checksum field like 'md5:abcd...' -> (algo, hex)."""
        if not checksum_field:
            return None, None
        m = re.match(
            r"^(md5|sha1|sha224|sha256|sha384|sha512):([0-9A-Fa-f]+)$",
            checksum_field.strip(),
        )
        if not m:
            return None, None
        return m.group(1), m.group(2)

    def _verify_checksum(self, data: bytes, algo: str, expected_hex: str) -> bool:
        """Verify bytes against expected hex checksum using given algo."""
        algo = algo.lower()
        if algo in {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"}:
            h = hashlib.new(algo)
            h.update(data)
            return h.hexdigest().lower() == expected_hex.lower()
        return False

    def _normalize_version(self, v: str) -> str:
        """Normalize version labels like 'v0.0.5' -> '0.0.5' for comparison."""
        v = str(v).strip()
        if v.lower().startswith("v"):
            v = v[1:]
        return v
