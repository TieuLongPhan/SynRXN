"""
synrxn.data.data_loader

DataLoader with support for:
 - Zenodo records (also datasets inside attached archives)
 - GitHub tags
 - GitHub commit SHA (including version='latest' resolution)

Key behaviour:
 - If a dataset is inside an attached archive (zip/tar*), DataLoader will
   download the archive, extract the member file and load it.
 - Cached zenodo .csv.gz files are verified with parse_checksum_field.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import io
import requests
import pandas as pd
from difflib import get_close_matches
import hashlib

from .constants import CONCEPT_DOI, GH_OWNER, GH_REPO
from .zenodo_client import ZenodoClient
from .github_client import GitHubClient
from .utils import (
    normalize_version,
    load_json_silent,
    save_json_silent,
    parse_checksum_field,
)


class DataLoader:
    """
    DataLoader for SynRXN data stored under ``Data/<task>/<name>.csv(.gz)``.

    The loader supports three sources:
      - ``'zenodo'``: pulls files from the Zenodo record for the project's concept DOI
      - ``'github'``: pulls files from a GitHub release tag or branch
      - ``'commit'``: pulls files from a specific commit SHA (``version='latest'`` resolves the tip SHA)

    :param task: Subfolder name under the repository's `Data/` directory (e.g. ``"class"``, ``"aam"``).
    :type task: str
    :param version:
        Source-dependent version identifier:

        - If ``source=='zenodo'``: Zenodo version string (e.g. ``"0.0.6"``) or ``None`` for the latest record.
        - If ``source=='github'``: GitHub release tag (e.g. ``"v0.0.6"`` or ``"0.0.6"``). ``'latest'`` can be used
          and will resolve to the latest release tag (when available) and otherwise fall back to a branch ref.
        - If ``source=='commit'``: commit SHA (40-char) or the string ``'latest'`` to resolve the tip SHA of a branch.
    :type version: Optional[str]
    :param cache_dir:
        Directory used to cache downloaded gz payloads, Zenodo record indices, and GitHub latest lookups.
        Defaults to ``~/.cache/synrxn``.
    :type cache_dir: Optional[pathlib.Path]
    :param timeout: HTTP timeout in seconds for requests. Default is ``20``.
    :type timeout: int
    :param user_agent: User-Agent string for outbound HTTP requests.
    :type user_agent: str
    :param max_workers: Maximum worker threads for :meth:`load_many`. Default ``6``.
    :type max_workers: int
    :param gh_ref:
        Optional explicit GitHub ref (branch name) used when resolving ``latest`` (commit or release fallback).
        If omitted, the repo default branch is used.
    :type gh_ref: Optional[str]
    :param gh_enable:
        If True enables GitHub-based retrieval. Required when ``source`` is ``'github'`` or ``'commit'``.
    :type gh_enable: bool
    :param source:
        One of ``{'zenodo', 'github', 'commit'}``.
    :type source: str
    :param resolve_on_init:
        If True and ``source=='zenodo'``, resolves the Zenodo record and file index during initialization.
    :type resolve_on_init: bool
    :param verify_checksum:
        If True, verifies Zenodo-provided checksums for downloads and cached payloads.
    :type verify_checksum: bool
    :param cache_record_index:
        If True, the Zenodo record file index will be cached in ``cache_dir``.
    :type cache_record_index: bool
    :param force_record_id:
        If provided, this numeric Zenodo record id will be used directly (bypassing version lookup).
    :type force_record_id: Optional[int]

    :raises ValueError:
        If ``source`` is not one of the supported values, or if GitHub retrieval
        is requested but ``gh_enable`` is False.
    :raises RuntimeError:
        If resolving ``version='latest'`` for ``source='commit'`` fails to return a commit SHA.

    Examples
    --------
    Zenodo (specific version):

    .. code-block:: python

        from synrxn.data import DataLoader
        from pathlib import Path

        dl = DataLoader(
            task="classification",
            source="zenodo",
            version="0.0.6",
            cache_dir=Path("~/.cache/synrxn").expanduser(),
        )
        print(dl.available_names())
        df = dl.load('schneider_b')
        print(df.head())

    GitHub (release tag):

    .. code-block:: python

        from synrxn.data import DataLoader
        from pathlib import Path

        dl = DataLoader(
            task="classification",
            source="github",
            version="v0.0.6",
            gh_enable=True,
            cache_dir=Path("~/.cache/synrxn").expanduser(),
        )
        print(dl.available_names())
        df = dl.load('schneider_b')
        print(df.head())

    Commit (explicit commit SHA):

    .. code-block:: python

        from synrxn.data import DataLoader
        from pathlib import Path

        dl = DataLoader(
            task="classification",
            source="commit",
            version="3e1612e2199e8b0e369fce3ed9aff3dda68e4c32",
            gh_enable=True,
            cache_dir=Path("~/.cache/synrxn").expanduser(),
        )
        print(dl.available_names())
        df = dl.load('schneider_b')
        print(df.head())

    Commit (resolve latest tip of a branch):

    .. code-block:: python

        from synrxn.data import DataLoader
        from pathlib import Path

        dl = DataLoader(
            task="classification",
            source="commit",
            version="latest",
            gh_enable=True,
            gh_ref="main",
            cache_dir=Path("~/.cache/synrxn").expanduser(),
        )
        # dl.version will be replaced with the resolved 40-char SHA
        print("resolved sha:", dl.version)
        print(dl.available_names())
        df = dl.load('schneider_b')
        print(df.head())

    Notes
    -----
    - ``version='latest'`` (commit or release) is non-deterministic; record the resolved value
      (``dl.version``) if you need reproducible results.
    - For heavy GitHub API usage, provide an authenticated session
    (add an Authorization token to ``dl._session.headers``).
    """

    def __init__(
        self,
        task: str,
        version: Optional[str] = None,
        cache_dir: Optional[Path] = Path("~/.cache/synrxn").expanduser(),
        timeout: int = 20,
        user_agent: str = "SynRXN-DataLoader/2.0",
        max_workers: int = 6,
        gh_ref: Optional[str] = None,
        gh_enable: bool = False,
        source: str = "zenodo",
        resolve_on_init: bool = False,
        verify_checksum: bool = True,
        cache_record_index: bool = True,
        force_record_id: Optional[int] = None,
    ) -> None:
        # normalize common short aliases to canonical folder names
        _alias_map = {
            "class": "classification",
            "prop": "property",
            "syn": "synthesis",
        }

        task_str = str(task).strip()
        task_clean = task_str.lower().strip("/\\ ")

        self.task = _alias_map.get(task_clean, task_clean)
        self.task = self.task.strip("/\\ ")

        self.version = version.strip() if isinstance(version, str) else None
        self.timeout = int(timeout)
        self.headers = {"User-Agent": user_agent}
        self.max_workers = int(max_workers)
        self.verify_checksum = bool(verify_checksum)

        if source not in {"zenodo", "github", "commit"}:
            raise ValueError("source must be one of {'zenodo', 'github', 'commit'}")
        self.source = source

        # HTTP session with small retry policy
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(
                total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504]
            )
            self._session.mount("https://", HTTPAdapter(max_retries=retry))
        except Exception:
            pass

        # cache dir
        self.cache_dir: Optional[Path] = (
            Path(cache_dir).expanduser().resolve() if cache_dir else None
        )
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # Zenodo client
        self._zenodo = ZenodoClient(
            session=self._session,
            cache_dir=self.cache_dir,
            cache_record_index=cache_record_index,
            timeout=self.timeout,
        )

        # GitHub options
        self.gh_enable = bool(gh_enable)
        if self.source in {"github", "commit"} and not self.gh_enable:
            raise ValueError("source='github' or 'commit' requires gh_enable=True")

        self._explicit_gh_ref = gh_ref
        self._gh_refs: List[Tuple[str, str]] = []
        self._github: Optional[GitHubClient] = None

        # Zenodo lazy state
        self._record_id: Optional[int] = (
            int(force_record_id) if force_record_id is not None else None
        )
        self._file_index: Dict[str, Dict] = {}

        self._names_cache_zenodo: Optional[List[str]] = None
        self._names_cache_github: Optional[List[str]] = None

        # Resolve 'latest' commit if requested
        if self.source == "commit":
            if not self.gh_enable:
                raise ValueError("source='commit' requires gh_enable=True")
            if not self.version or self.version.lower() in {"latest", "head", "tip"}:
                resolved = self._resolve_latest_commit_sha(
                    branch=self._explicit_gh_ref, force_refresh=False
                )
                if resolved is None:
                    raise RuntimeError(
                        "Failed to resolve latest commit SHA from GitHub for repository "
                        f"{GH_OWNER}/{GH_REPO}"
                    )
                self.version = resolved
                self._gh_refs = [("commit", resolved)]
            else:
                self._gh_refs = [("commit", self.version)]

        # GitHub tag handling for source == 'github'
        if self.source == "github":
            if self.version:
                norm = normalize_version(self.version)
                self._gh_refs = [("tag", f"v{norm}"), ("tag", norm)]
            else:
                self._gh_refs = [("branch", self._explicit_gh_ref or "main")]

        # Create GitHub client if enabled
        if self.gh_enable:
            if not self._gh_refs:
                ref = self._explicit_gh_ref or "main"
                self._gh_refs = [("branch", ref)]
            self._github = GitHubClient(
                session=self._session,
                timeout=self.timeout,
                owner=GH_OWNER,
                repo=GH_REPO,
                ref_candidates=self._gh_refs,
            )

        if resolve_on_init and self.source == "zenodo":
            self._ensure_record_resolved()

    def __del__(self):
        try:
            self._session.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"DataLoader(task={self.task!r}, version={self.version!r}, record={self._record_id}, "
            f"source={self.source!r}, gh_refs={self._gh_refs}, cache_dir={self.cache_dir})"
        )

    # ------------------ utilities ------------------
    def _normalize_basename(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        n = name.strip()
        prefixes = [
            f"Data/{self.task}/",
            f"data/{self.task}/",
            f"{self.task}/",
            f"/{self.task}/",
            "Data/",
            "data/",
        ]
        for p in prefixes:
            if n.startswith(p):
                n = n[len(p) :]  # noqa
                break
        n = n.strip("/\\ ").strip()
        lower = n.lower()
        if lower.endswith(".csv.gz"):
            n = n[: -len(".csv.gz")]
        elif lower.endswith(".csv"):
            n = n[: -len(".csv")]
        return n.strip()

    def _maybe_set_pyarrow(self, pd_kw: Dict) -> None:
        if "engine" in pd_kw:
            return
        try:
            import pyarrow  # noqa: F401

            pd_kw["engine"] = "pyarrow"
        except Exception:
            pass

    # ------------------ GitHub latest resolution ------------------
    def _github_latest_cache_path(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return (self.cache_dir / "github_latest_cache.json").resolve()

    def _resolve_latest_commit_sha(
        self, branch: Optional[str] = None, force_refresh: bool = False
    ) -> Optional[str]:
        if not self.gh_enable:
            return None

        cache_path = self._github_latest_cache_path()
        cache = load_json_silent(cache_path) if cache_path else {}
        cache_key = branch or "__default__"
        if not force_refresh:
            cached = cache.get(cache_key)
            if cached and isinstance(cached, dict):
                sha = cached.get("sha")
                if sha:
                    return sha

        br = branch
        if not br:
            try:
                repo_url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}"
                r = self._session.get(repo_url, timeout=self.timeout)
                r.raise_for_status()
                br = r.json().get("default_branch") or "main"
            except Exception:
                br = "main"

        try:
            commits_url = (
                f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/commits/{br}"
            )
            r = self._session.get(commits_url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            sha = data.get("sha") or (
                data[0].get("sha") if isinstance(data, list) and data else None
            )
            if sha:
                try:
                    cache[cache_key] = {"sha": sha}
                    if cache_path:
                        save_json_silent(cache_path, cache)
                except Exception:
                    pass
                return sha
        except Exception:
            pass
        return None

    # ------------------ Zenodo helpers ------------------
    def _ensure_record_resolved(self, force_refresh: bool = False) -> None:
        if self.source != "zenodo":
            return
        if self._record_id is None:
            self._record_id = self._zenodo.resolve_record_id(CONCEPT_DOI, self.version)
        if force_refresh or not self._file_index:
            self._file_index = self._zenodo.build_file_index(self._record_id)
            self._names_cache_zenodo = None

    def find_zenodo_keys(self, term: str) -> List[str]:
        """
        Public helper to search keys in the currently-loaded Zenodo file index.

        Resolves the record index if not already loaded. Returns an empty list if
        no client/index is available or if an error occurs.
        """
        try:
            if not self._file_index:
                # best-effort: attempt to resolve the record (only when source is zenodo)
                try:
                    self._ensure_record_resolved()
                except Exception:
                    pass
            return self._zenodo.find_keys(self._file_index, term)
        except Exception:
            return []

    def available_names(self, refresh: bool = False) -> List[str]:
        if self.source == "zenodo":
            self._ensure_record_resolved(force_refresh=refresh)
            raw_names = self._zenodo.available_names(
                task=self.task,
                record_id=self._record_id or 0,
                file_index=self._file_index,
                include_archives=True,
            )
            names = sorted(
                {self._normalize_basename(n) for n in raw_names if isinstance(n, str)}
            )
            self._names_cache_zenodo = names
            return list(names)
        else:
            if not self.gh_enable:
                return []
            if self._names_cache_github is not None and not refresh:
                return list(self._names_cache_github)
            if not self._github:
                self._github = GitHubClient(
                    session=self._session,
                    timeout=self.timeout,
                    owner=GH_OWNER,
                    repo=GH_REPO,
                    ref_candidates=self._gh_refs,
                )
            raw = self._github.list_names(self.task)
            names = sorted(
                {self._normalize_basename(n) for n in raw if isinstance(n, str)}
            )
            self._names_cache_github = names
            return list(names)

    # ------------------ checksum helper ------------------
    def _compute_hex(self, data: bytes, algo: str) -> str:
        try:
            h = hashlib.new(algo)
        except Exception:
            h = hashlib.sha256()
        h.update(data)
        return h.hexdigest().lower()

    # ------------------ core loader ------------------
    def load(
        self,
        name: str,
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        **pd_kw,
    ) -> pd.DataFrame:
        self._maybe_set_pyarrow(pd_kw)
        name = self._normalize_basename(name)

        rel_gz = f"Data/{self.task}/{name}.csv.gz"
        rel_csv = f"Data/{self.task}/{name}.csv"

        tried: List[str] = []
        last_err = None

        def _read_bytes(content: bytes, gz: bool) -> pd.DataFrame:
            buf = io.BytesIO(content)
            return pd.read_csv(
                buf, compression=("gzip" if gz else None), dtype=dtype, **pd_kw
            )

        # --- 1) exact keys on Zenodo ---
        def _try_exact_zenodo() -> Optional[pd.DataFrame]:
            nonlocal last_err
            self._ensure_record_resolved()

            cache_path = (
                (self.cache_dir / f"{self.task}__{name}.csv.gz")
                if (self.cache_dir)
                else None
            )

            for key in (rel_gz, rel_csv):
                if key in self._file_index:
                    meta = self._file_index[key]
                    # verify cached gz if present and meta has checksum
                    if (
                        key.endswith(".csv.gz")
                        and use_cache
                        and cache_path
                        and cache_path.exists()
                    ):
                        algo, expected_hex = parse_checksum_field(
                            meta.get("checksum", "") or ""
                        )
                        if algo and expected_hex:
                            try:
                                cached_bytes = cache_path.read_bytes()
                                got_hex = self._compute_hex(cached_bytes, algo)
                                if got_hex.lower() == expected_hex.lower():
                                    return _read_bytes(cached_bytes, gz=True)
                                else:
                                    try:
                                        cache_path.unlink()
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    # download via ZenodoClient (stream_to_temp_and_verify will check checksum)
                    resp = self._zenodo.get_download_response(
                        meta, self._record_id or 0
                    )
                    if resp is not None:
                        tried.append(resp.url)
                        temp_path = None
                        try:
                            suffix = ".gz" if key.endswith(".gz") else ".csv"
                            temp_path = self._zenodo.stream_to_temp_and_verify(
                                resp, meta, suffix=suffix
                            )
                            content = Path(temp_path).read_bytes()
                            if use_cache and self.cache_dir and key.endswith(".csv.gz"):
                                try:
                                    (
                                        self.cache_dir / f"{self.task}__{name}.csv.gz"
                                    ).write_bytes(content)
                                except Exception:
                                    pass
                            return _read_bytes(content, gz=key.endswith(".csv.gz"))
                        except Exception as e:
                            last_err = e
                        finally:
                            try:
                                if temp_path and Path(temp_path).exists():
                                    Path(temp_path).unlink()
                            except Exception:
                                pass
                    else:
                        tried.append(
                            f"(no usable download candidate from Zenodo metadata for {key})"
                        )
            return None

        # --- 2) fuzzy: check direct keys (non-archive) ---
        def _try_fuzzy_zenodo() -> Optional[pd.DataFrame]:
            nonlocal last_err
            # look for candidate keys in record index matching name/task patterns
            candidates = (
                self._zenodo.find_keys(self._file_index, f"{self.task}/{name}")
                + self._zenodo.find_keys(self._file_index, f"{self.task}_{name}")
                + self._zenodo.find_keys(self._file_index, name)
            )
            seen = []
            for c in candidates:
                if c not in seen:
                    seen.append(c)
            for key in seen:
                if not (key.endswith(".csv") or key.endswith(".csv.gz")):
                    continue
                meta = self._file_index.get(key, {})
                resp = self._zenodo.get_download_response(meta, self._record_id or 0)
                if resp is None:
                    tried.append(
                        f"(no usable download candidate from Zenodo metadata for {key})"
                    )
                    last_err = RuntimeError(
                        f"No usable download candidate for Zenodo key {key}"
                    )
                    continue
                tried.append(resp.url)
                temp_path = None
                try:
                    suffix = ".gz" if key.endswith(".gz") else ".csv"
                    temp_path = self._zenodo.stream_to_temp_and_verify(
                        resp, meta, suffix=suffix
                    )
                    content = Path(temp_path).read_bytes()
                    if use_cache and self.cache_dir and key.endswith(".csv.gz"):
                        try:
                            (
                                self.cache_dir / f"{self.task}__{name}.csv.gz"
                            ).write_bytes(content)
                        except Exception:
                            pass
                    return _read_bytes(content, gz=key.endswith(".csv.gz"))
                except Exception as e:
                    last_err = e
                finally:
                    try:
                        if temp_path and Path(temp_path).exists():
                            Path(temp_path).unlink()
                    except Exception:
                        pass
            return None

        # --- 3) archive members: inspect archives and extract member if present ---
        def _try_archive_members() -> Optional[pd.DataFrame]:
            nonlocal last_err
            # find archive keys in record index
            archive_keys = [
                k
                for k in self._file_index.keys()
                if k.lower().endswith((".zip", ".tgz", ".tar.gz", ".tar"))
            ]
            if not archive_keys:
                return None

            # build possible member tails to match (lowered)
            candidates_tail = [
                f"data/{self.task}/{name}.csv.gz",
                f"data/{self.task}/{name}.csv",
                f"data/{self.task}/{name.replace('_','-')}.csv.gz",
                f"data/{self.task}/{name.replace('_','-')}.csv",
                f"data/{self.task}/{name.replace('-','_')}.csv.gz",
                f"data/{self.task}/{name.replace('-','_')}.csv",
                f"{self.task}/{name}.csv.gz",
                f"{self.task}/{name}.csv",
                f"{name}.csv.gz",
                f"{name}.csv",
            ]

            # search archive member listings (cached) first
            for ak in archive_keys:
                meta = self._file_index.get(ak, {})
                try:
                    members = self._zenodo.list_archive_members_cached(
                        self._record_id or 0, ak, meta
                    )
                except Exception:
                    members = []
                if not members:
                    continue
                for m in members:
                    ml = m.replace("\\", "/").lower()
                    for tail in candidates_tail:
                        if ml.endswith(tail):
                            # found candidate member inside archive
                            try:
                                # download archive (stream_to_temp_and_verify will verify checksum)
                                resp = self._zenodo.get_download_response(
                                    meta, self._record_id or 0
                                )
                                if resp is None:
                                    tried.append(
                                        f"(no usable download candidate for archive {ak})"
                                    )
                                    continue
                                tried.append(resp.url)
                                temp_path = None
                                try:
                                    suffix = (
                                        ".zip"
                                        if ak.lower().endswith(".zip")
                                        else ".tar"
                                    )
                                    temp_path = self._zenodo.stream_to_temp_and_verify(
                                        resp, meta, suffix=suffix
                                    )
                                    # extract bytes for member m
                                    member_bytes = self._zenodo.extract_member_bytes(
                                        Path(temp_path), m
                                    )
                                    if member_bytes is None:
                                        last_err = RuntimeError(
                                            f"Member {m} not found in archive after download"
                                        )
                                        continue
                                    # if the member name endswith .csv.gz, it's gzipped inside archive
                                    is_gz = m.lower().endswith(".csv.gz")
                                    # optionally cache extracted gz to cache_dir for future
                                    if is_gz and use_cache and self.cache_dir:
                                        try:
                                            (
                                                self.cache_dir
                                                / f"{self.task}__{name}.csv.gz"
                                            ).write_bytes(member_bytes)
                                        except Exception:
                                            pass
                                    return _read_bytes(member_bytes, gz=is_gz)
                                finally:
                                    try:
                                        if temp_path and Path(temp_path).exists():
                                            Path(temp_path).unlink()
                                    except Exception:
                                        pass
                            except Exception as e:
                                last_err = e
                                continue
            return None

        # --- 4) GitHub-like retrieval ---
        def _try_github_like() -> Optional[pd.DataFrame]:
            nonlocal last_err
            if not self.gh_enable:
                return None
            if not self._github:
                self._github = GitHubClient(
                    session=self._session,
                    timeout=self.timeout,
                    owner=GH_OWNER,
                    repo=GH_REPO,
                    ref_candidates=self._gh_refs,
                )
            for ext in (".csv.gz", ".csv"):
                url = self._github.raw_url(self.task, name, ext)
                if not url:
                    continue
                tried.append(url)
                try:
                    r = self._session.get(url, timeout=self.timeout, stream=True)
                    if r.status_code == 200:
                        content = r.content
                        if ext == ".csv.gz" and use_cache and self.cache_dir:
                            try:
                                (
                                    self.cache_dir / f"{self.task}__{name}.csv.gz"
                                ).write_bytes(content)
                            except Exception:
                                pass
                        return _read_bytes(content, gz=(ext == ".csv.gz"))
                    else:
                        last_err = RuntimeError(f"HTTP {r.status_code} for {url}")
                except Exception as e:
                    last_err = e
            return None

        # Try strategies in order
        if self.source == "zenodo":
            for fn in (_try_exact_zenodo, _try_fuzzy_zenodo, _try_archive_members):
                df = fn()
                if df is not None:
                    return df
        else:
            df = _try_github_like()
            if df is not None:
                return df

        # Not found: diagnostics
        zenodo_keys = []
        try:
            if self._file_index:
                zenodo_keys = sorted(self._file_index.keys())
        except Exception:
            zenodo_keys = []

        github_names = []
        try:
            if self.gh_enable and self.source in {"github", "commit"}:
                github_names = self.available_names(refresh=True)
        except Exception:
            github_names = []

        canonical_candidates = [
            f"Data/{self.task}/{name}.csv.gz",
            f"Data/{self.task}/{name}.csv",
            f"Data/{self.task}_{name}.csv.gz",
            f"Data/{self.task}_{name}.csv",
            f"{name}.csv.gz",
            f"{name}.csv",
        ]
        if "-" in name:
            canonical_candidates += [
                f"Data/{self.task}/{name.replace('-', '_')}.csv.gz",
                f"Data/{self.task}/{name.replace('-', '_')}.csv",
            ]
        if "_" in name:
            canonical_candidates += [
                f"Data/{self.task}/{name.replace('_', '-')}.csv.gz",
                f"Data/{self.task}/{name.replace('_', '-')}.csv",
            ]

        msg_lines = [
            f"Failed to fetch dataset '{name}' for task '{self.task}'.",
            f"Concept DOI: {CONCEPT_DOI}",
            f"Version: {self.version or 'latest'} (record {self._record_id})",
            f"Source: {self.source}",
            "",
            "Tried URLs:",
        ]
        if tried:
            msg_lines += [f"  {u}" for u in tried]
        else:
            msg_lines += ["  (no candidate URLs were attempted)"]

        msg_lines += ["", "Canonical candidate file names we would look for:"]
        msg_lines += [f"  {c}" for c in canonical_candidates]

        if zenodo_keys and self.source == "zenodo":
            msg_lines += [
                "",
                f"Zenodo record {self._record_id} contains these file keys ({len(zenodo_keys)}):",
            ]
            msg_lines += [f"  {k}" for k in zenodo_keys]

        if github_names and self.source in {"github", "commit"}:
            msg_lines += [
                "",
                "GitHub Data/ filenames discovered for this ref:",
            ]
            msg_lines += [f"  {n}" for n in github_names]

        try:
            avail = self.available_names(refresh=False)
        except Exception:
            avail = []
        if avail:
            suggestions = get_close_matches(name, avail, n=5, cutoff=0.4)
            if suggestions:
                msg_lines += ["", f"Did you mean: {suggestions} ?"]

        if last_err:
            msg_lines += ["", f"Last error: {last_err!s}"]

        raise FileNotFoundError("\n".join(msg_lines))

    # ------------------ batch loader ------------------
    def load_many(
        self,
        names: Iterable[str],
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        parallel: bool = True,
        **pd_kw,
    ) -> Dict[str, pd.DataFrame]:
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
