"""
synrxn.data.data_loader

High-level DataLoader for SynRXN datasets with explicit source selection,
version->record caching (via ZenodoClient), archive inspection (via ZenodoClient),
and GitHub mirror support (via GitHubClient).

This file assumes `ZenodoClient` and `GitHubClient` live in the same package
(`synrxn.data.zenodo_client`, `synrxn.data.github_client`) and that shared
helpers exist in `synrxn.data.utils`.

:features:
  - source='zenodo' | 'github' | 'any'
  - inspects CSVs inside attached ZIP/TAR archives (cached)
  - stream hashing and 1 MiB chunks for IO
  - prefers the pyarrow CSV engine when available
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote as urlquote
import io
import os
import tempfile
import requests
import pandas as pd

from .constants import CONCEPT_DOI, GH_OWNER, GH_REPO
from .zenodo_client import ZenodoClient
from .github_client import GitHubClient
from .utils import normalize_version, parse_checksum_field


class DataLoader:
    """
    Load CSV(.gz) datasets stored under `Data/<task>/<name>.csv(.gz)`.

    :param task: Subfolder under `Data/` (e.g., "aam", "rbl", "class", "prop", "synthesis").
    :param version: Version label to pin (e.g., "0.0.5" or "v0.0.5"). If None, uses latest under the concept DOI.
    :param cache_dir: Optional path to cache Zenodo record indices, version map, downloaded `.csv.gz` payloads,
                      and archive member lists (all handled in clients).
    :param timeout: HTTP timeout (seconds). Default: 20.
    :param user_agent: HTTP User-Agent header. Default: "SynRXN-DataLoader/2.0".
    :param max_workers: Max threads for load_many. Default: 6.
    :param gh_ref: Optional explicit GitHub ref for listing/reads. If None and version exists, tries
                   "v{version}", "{version}", then "main".
    :param gh_enable: Whether GitHub network calls are permitted. Must be True for source='github' or to use GH in 'any'.
    :param source: One of {"zenodo", "github", "any"}; default "zenodo".
    :param resolve_on_init: If True, resolve Zenodo record id and file index during __init__ (may incur network).
    :param verify_checksum: Verify Zenodo-provided checksums when available (handled in client). Default True.
    :param cache_record_index: Persist Zenodo record file index to cache_dir (handled in client). Default True.
    :param force_record_id: (Discouraged) Provide a fixed Zenodo record id to bypass version lookup.

    Examples
    --------
    The following examples are written in Sphinx-friendly rst style and demonstrate
    how to construct and use the loader with different `source` settings.

    Zenodo-first (recommended for reproducibility)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Resolve the Zenodo record for the configured ``version`` (provenance-first),
    list discovered datasets (including CSVs inside attached archives), and load one.

    .. code-block:: python

        from pathlib import Path
        from synrxn.data import DataLoader

        # Zenodo-first (recommended for reproducibility)
        ldr = DataLoader(
            task="rbl",
            version="0.0.5",
            source="zenodo",               # only use Zenodo
            cache_dir=Path("~/.cache/synrxn").expanduser(),
            gh_enable=False,               # ensure we don't hit GitHub
            resolve_on_init=True           # optional: resolve record & index on init
        )

        # print top-level Zenodo file keys (archives, zips, etc.)
        ldr.print_zenodo_files()

        # see discovered names (includes CSVs inside archives attached to the record)
        names = ldr.names
        print("Detected datasets:", names)

        # load a single dataset (raises FileNotFoundError with helpful diagnostics if missing)
        try:
            df = ldr.load("mis")          # loads Data/rbl/mis.csv(.gz)
        except FileNotFoundError as e:
            print("Load failed:", e)
        else:
            print(df.shape)
            print(df.head())

    GitHub-only usage (fast mirror)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If you want to bypass Zenodo and pull directly from the repository (useful for development),
    set ``source='github'`` and enable GitHub calls.

    .. code-block:: python

        from synrxn.data import DataLoader
        from pathlib import Path

        gh = DataLoader(
            task="rbl",
            source="github",               # only use GitHub
            gh_enable=True,                # permit GitHub network calls
            gh_ref="main",                 # explicit branch/tag/ref to try
            cache_dir=Path("~/.cache/synrxn").expanduser()
        )

        # list names from the GitHub Data/<task>/ folder (first successful ref)
        print("GitHub names:", gh.names)

        # load (tries raw.githubusercontent.com)
        df = gh.load("schneider_rbl")
        print(df.columns)

    Mixed strategy: zenodo -> github -> archives
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Use ``source='any'`` to try Zenodo direct files first, then GitHub, then Zenodo archives.

    .. code-block:: python

        dl = DataLoader(task="rbl", version="0.0.5", source="any", gh_enable=True,
                        cache_dir=Path("~/.cache/synrxn").expanduser())

        # load_many uses a threadpool by default (max_workers from constructor)
        dfs = dl.load_many(["mis", "schneider_rbl"])

        # iterate
        for name, df in dfs.items():
            print(name, df.shape)

    Error handling and diagnostics
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When a dataset is not found the loader raises ``FileNotFoundError`` with:
    * attempted candidate URLs,
    * canonical candidate filenames,
    * Zenodo record file keys (and archive member names when available),
    * GitHub filenames (if enabled),
    * close-match suggestions.

    .. code-block:: python

        try:
            df = dl.load("not_a_dataset")
        except FileNotFoundError as exc:
            # helpful, multi-line diagnostic for debugging
            print(str(exc))

    Caching notes
    ~~~~~~~~~~~~~
    * Provide ``cache_dir`` to persist:
      - the Zenodo record index (file index & version->record mapping),
      - archive member listings (so we do not re-download archives just to inspect them),
      - downloaded ``*.csv.gz`` payloads (when available).
    * ``resolve_on_init=True`` will populate the record index on construction (useful in scripts).
    """

    def __init__(
        self,
        task: str,
        version: Optional[str] = None,
        cache_dir: Optional[Path] = None,
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
        self.task = str(task).strip("/")
        self.version = version.strip() if isinstance(version, str) else None
        self.timeout = int(timeout)
        self.headers = {"User-Agent": user_agent}
        self.max_workers = int(max_workers)
        self.verify_checksum = bool(verify_checksum)

        if source not in {"zenodo", "github", "any"}:
            raise ValueError("source must be one of {'zenodo', 'github', 'any'}")
        self.source = source

        # session with light retries
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
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # clients
        self._zenodo = ZenodoClient(
            session=self._session,
            cache_dir=self.cache_dir,
            cache_record_index=cache_record_index,
            timeout=self.timeout,
        )

        self.gh_enable = bool(gh_enable)
        if self.source == "github" and not self.gh_enable:
            raise ValueError("source='github' requires gh_enable=True")

        self._gh_refs: List[Tuple[str, str]] = []
        if self.gh_enable and self.source in {"github", "any"}:
            if gh_ref:
                self._gh_refs = [("heads", gh_ref)]
            elif self.version:
                norm = normalize_version(self.version)
                self._gh_refs = [
                    ("tags", f"v{norm}"),
                    ("tags", norm),
                    ("heads", "main"),
                ]
            else:
                self._gh_refs = [("heads", "main")]
        self._github = GitHubClient(
            session=self._session,
            timeout=self.timeout,
            owner=GH_OWNER,
            repo=GH_REPO,
            ref_candidates=self._gh_refs,
        )

        # lazy Zenodo state
        self._record_id: Optional[int] = (
            int(force_record_id) if force_record_id is not None else None
        )
        self._file_index: Dict[str, Dict] = {}

        # caches for name listings
        self._names_cache_zenodo: Optional[List[str]] = None
        self._names_cache_github: Optional[List[str]] = None

        if resolve_on_init and self.source in {"zenodo", "any"}:
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

    # ---------- Debug helper ----------
    def print_zenodo_files(self, limit: int = 200) -> None:
        """
        Print visible file keys attached to the resolved Zenodo record.

        :param limit: Max entries to print.
        """
        if self.source == "github":
            print("(Zenodo not in use for source='github')")
            return
        self._ensure_record_resolved()
        keys = list(self._file_index.keys())
        print(f"Zenodo record {self._record_id} has {len(keys)} file(s).")
        for k in keys[:limit]:
            print(" ", k)
        if len(keys) > limit:
            print("  ... (remaining files elided)")

    # ---------- Listing ----------
    @property
    def names(self) -> List[str]:
        """Alias for available_names()."""
        return self.available_names()

    def available_names(self, refresh: bool = False) -> List[str]:
        """
        List dataset base names discovered in the selected source(s).

        :param refresh: If True, re-fetch and rebuild caches.
        :return: Sorted list of base names without extensions.
        """
        z_names: List[str] = []
        g_names: List[str] = []
        if self.source in {"zenodo", "any"}:
            self._ensure_record_resolved(force_refresh=refresh)
            z_names = self._available_names_zenodo(refresh=refresh)
        if self.source in {"github", "any"} and self.gh_enable:
            g_names = self._available_names_github(refresh=refresh)
        if self.source == "zenodo":
            return z_names
        if self.source == "github":
            return g_names
        return sorted(set(z_names).union(g_names))

    def _available_names_zenodo(self, refresh: bool = False) -> List[str]:
        if self._names_cache_zenodo is not None and not refresh:
            return list(self._names_cache_zenodo)
        names = self._zenodo.available_names(
            task=self.task,
            record_id=self._record_id or 0,
            file_index=self._file_index,
            include_archives=True,
        )
        self._names_cache_zenodo = names
        return list(self._names_cache_zenodo)

    def _available_names_github(self, refresh: bool = False) -> List[str]:
        if not self.gh_enable:
            return []
        if self._names_cache_github is not None and not refresh:
            return list(self._names_cache_github)
        self._names_cache_github = self._github.list_names(self.task)
        return list(self._names_cache_github)

    # ---------- Core loading ----------
    def _ensure_record_resolved(self, force_refresh: bool = False) -> None:
        if self.source == "github":
            return
        if self._record_id is None:
            self._record_id = self._zenodo.resolve_record_id(CONCEPT_DOI, self.version)
        if force_refresh or not self._file_index:
            self._file_index = self._zenodo.build_file_index(self._record_id)
            self._names_cache_zenodo = None  # invalidate names cache

    def _maybe_set_pyarrow(self, pd_kw: Dict) -> None:
        if "engine" in pd_kw:
            return
        try:
            import pyarrow  # noqa: F401

            pd_kw["engine"] = "pyarrow"
        except Exception:
            pass

    def find_zenodo_keys(self, term: str) -> List[str]:
        """
        Case-insensitive search over the Zenodo record file keys.

        :param term: Substring to search for (e.g., "rbl/mis" or "mis").
        :return: List of matching keys.
        """
        if self.source == "github":
            return []
        self._ensure_record_resolved()
        return self._zenodo.find_keys(self._file_index, term)

    def load(
        self,
        name: str,
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        **pd_kw,
    ) -> pd.DataFrame:
        """
        Load `Data/<task>/<name>.csv(.gz)` according to `source`.

        :param name: Base dataset name (no extension).
        :param use_cache: Persist gz payloads to cache_dir if True.
        :param dtype: Optional pandas dtype mapping.
        :param pd_kw: Extra pandas.read_csv kwargs.
        :return: pandas.DataFrame
        :raises FileNotFoundError: if dataset not found in selected source(s).
        """
        self._maybe_set_pyarrow(pd_kw)

        rel_gz = f"Data/{self.task}/{name}.csv.gz"
        rel_csv = f"Data/{self.task}/{name}.csv"
        tried: List[str] = []
        last_err = None

        def _read_bytes(content: bytes, gz: bool) -> pd.DataFrame:
            buf = io.BytesIO(content)
            return pd.read_csv(
                buf, compression=("gzip" if gz else None), dtype=dtype, **pd_kw
            )

        # ---- strategies (each returns DataFrame or None) ----
        def _try_zenodo_files() -> Optional[pd.DataFrame]:
            nonlocal last_err
            self._ensure_record_resolved()

            # exact keys
            for key in (rel_gz, rel_csv):
                if key in self._file_index:
                    meta = self._file_index[key]
                    resp = self._zenodo.get_download_response(
                        meta, self._record_id or 0
                    )
                    if resp is not None:
                        tried.append(resp.url)
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

            # fuzzy candidates within record index
            candidates = (
                self.find_zenodo_keys(f"{self.task}/{name}")
                + self.find_zenodo_keys(f"{self.task}_{name}")
                + self.find_zenodo_keys(name)
            )
            seen: List[str] = []
            for c in candidates:
                if c not in seen:
                    seen.append(c)
            for key in seen:
                if not (key.endswith(".csv") or key.endswith(".csv.gz")):
                    continue
                meta = self._file_index[key]
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

        def _try_github() -> Optional[pd.DataFrame]:
            nonlocal last_err
            if not self.gh_enable or self.source not in {"github", "any"}:
                return None
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

        def _try_zenodo_archives() -> Optional[pd.DataFrame]:
            nonlocal last_err
            self._ensure_record_resolved()
            archive_keys = [
                k
                for k in self._file_index.keys()
                if k.lower().endswith((".zip", ".tar.gz", ".tgz", ".tar"))
            ]
            # candidates to look for inside archives
            inner_exact = [
                f"Data/{self.task}/{name}.csv.gz",
                f"Data/{self.task}/{name}.csv",
            ]

            for ak in archive_keys:
                meta = self._file_index[ak]
                resp = self._zenodo.get_download_response(meta, self._record_id or 0)
                if resp is None:
                    tried.append(f"(no usable download candidate for archive {ak})")
                    last_err = RuntimeError(
                        f"No usable download candidate for archive {ak}"
                    )
                    continue
                tried.append(resp.url)
                temp_path = None
                try:
                    suffix = ".zip" if ak.lower().endswith(".zip") else ".tar"
                    temp_path = self._zenodo.stream_to_temp_and_verify(
                        resp, meta, suffix=suffix
                    )

                    # First try exact members, else scan for loose match under Data/<task>/
                    members = self._zenodo.list_archive_members_cached(
                        self._record_id or 0, ak, meta
                    )
                    member = None
                    # exact
                    for ec in inner_exact:
                        if ec in members:
                            member = ec
                            break
                    # loose
                    if member is None:
                        low_members = [m.lower().replace("\\", "/") for m in members]
                        for idx, m in enumerate(low_members):
                            if (
                                f"data/{self.task}/{name}" in m
                                or f"data/{self.task}_{name}" in m
                                or name.lower() in m
                            ):
                                member = members[idx]
                                break

                    if member:
                        content = self._zenodo.extract_member_bytes(temp_path, member)
                        if content is not None:
                            gz = member.endswith(".gz")
                            if gz and use_cache and self.cache_dir:
                                try:
                                    (
                                        self.cache_dir / f"{self.task}__{name}.csv.gz"
                                    ).write_bytes(content)
                                except Exception:
                                    pass
                            return _read_bytes(content, gz=gz)
                except Exception as e:
                    last_err = e
                finally:
                    try:
                        if temp_path and Path(temp_path).exists():
                            Path(temp_path).unlink()
                    except Exception:
                        pass
            return None

        # Strategy
        if self.source == "zenodo":
            df = _try_zenodo_files()
            if df is not None:
                return df
            df = _try_zenodo_archives()
            if df is not None:
                return df
        elif self.source == "github":
            df = _try_github()
            if df is not None:
                return df
        else:  # any
            df = _try_zenodo_files()
            if df is not None:
                return df
            df = _try_github()
            if df is not None:
                return df
            df = _try_zenodo_archives()
            if df is not None:
                return df

        # -------- Not found — improved diagnostic --------
        zenodo_keys: List[str] = []
        try:
            if self._file_index:
                zenodo_keys = sorted(self._file_index.keys())
        except Exception:
            zenodo_keys = []

        github_names: List[str] = []
        try:
            if self.gh_enable and self.source in {"github", "any"}:
                github_names = self._available_names_github(refresh=True)
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
        seen = set()
        canonical_candidates = [
            c for c in canonical_candidates if not (c in seen or seen.add(c))
        ]

        msg_lines: List[str] = [
            f"Failed to fetch dataset '{name}' for task '{self.task}'.",
            f"Concept DOI: {CONCEPT_DOI}",
            f"Version: {self.version or 'latest'} (record {self._record_id})",
            f"Source: {self.source}",
            "",
            "Tried URLs / archives (in order attempted):",
        ]
        if tried:
            msg_lines += [f"  {u}" for u in tried]
        else:
            msg_lines += ["  (no candidate URLs/archives were attempted)"]

        msg_lines += ["", "Canonical candidate file names we would look for:"]
        msg_lines += [f"  {c}" for c in canonical_candidates]

        if zenodo_keys:
            msg_lines += [
                "",
                f"Zenodo record {self._record_id} contains these file keys ({len(zenodo_keys)}):",
            ]
            if len(zenodo_keys) > 300:
                msg_lines += [f"  {k}" for k in zenodo_keys[:200]]
                msg_lines += [f"  ... ({len(zenodo_keys)-200} more keys elided)"]
            else:
                msg_lines += [f"  {k}" for k in zenodo_keys]

        if github_names:
            msg_lines += [
                "",
                "GitHub Data/ filenames discovered for this task (from configured refs):",
            ]
            msg_lines += [f"  {n}" for n in github_names]

        # Suggestions
        avail = self.available_names(refresh=False)
        from difflib import get_close_matches

        suggestions = get_close_matches(name, avail, n=5, cutoff=0.4) if avail else []
        if suggestions:
            msg_lines += ["", f"Did you mean: {suggestions} ?"]

        if last_err:
            msg_lines += ["", f"Last error: {last_err!s}"]

        raise FileNotFoundError("\n".join(msg_lines))

    # ---------- Batch ----------
    def load_many(
        self,
        names: Iterable[str],
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        parallel: bool = True,
        **pd_kw,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load many datasets into a dict.

        :param names: Iterable of base names.
        :param use_cache: If True, cache gz payloads to disk.
        :param dtype: Optional dtype mapping for pandas.
        :param parallel: If True and max_workers>1, use ThreadPoolExecutor.
        :param pd_kw: Additional pandas.read_csv kwargs.
        :return: {name: DataFrame}
        :raises RuntimeError: On first failed load (with chained FileNotFoundError).
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
