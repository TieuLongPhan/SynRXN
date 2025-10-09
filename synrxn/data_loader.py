"""SynRXN DataLoader: object-oriented loader for datasets hosted in the SynRXN GitHub repo.

The loader fetches files from the public raw tree (default:
https://github.com/TieuLongPhan/SynRXN/raw/refs/heads/main/Data).
Files are expected under Data/<task>/<name>.csv(.gz).

Example
-------
>>> from synrxn.data_loader import DataLoader
>>> dl = DataLoader(task="aam")
>>> dl.print_names()
>>> df = dl.load("ecoli")
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import io
import difflib
import math
import requests
import pandas as pd

# Default raw base (public)
_RAW_BASE_TPL = "https://github.com/{owner}/{repo}/raw/refs/heads/{branch}/Data"
# Default GitHub Contents API base for listing files
_API_BASE_TPL = (
    "https://api.github.com/repos/{owner}/{repo}/contents/Data/{task}?ref={branch}"
)


class DataLoader:
    """
    Object-oriented loader for CSV(.gz) datasets stored in a GitHub repo Data/ tree.

    :param task: Subfolder under `Data/` (e.g. "aam", "rbl", "class", "prop", "synthesis").
    :param cache_dir: Optional path to cache downloaded gz bytes. If provided, cached files are
                      stored as ``{task}__{name}.csv.gz``.
    :param owner: GitHub owner (default: "TieuLongPhan").
    :param repo: GitHub repository name (default: "SynRXN").
    :param branch: Branch/ref to use (default: "main").
    :param timeout: HTTP request timeout in seconds (default: 20).
    :param user_agent: HTTP User-Agent header (default: "DataLoader/1.0").
    """

    def __init__(
        self,
        task: str,
        cache_dir: Optional[Path] = None,
        owner: str = "TieuLongPhan",
        repo: str = "SynRXN",
        branch: str = "main",
        timeout: int = 20,
        user_agent: str = "DataLoader/1.0",
        max_workers: int = 6,
    ) -> None:
        self.task = str(task).strip("/")
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.timeout = int(timeout)
        self.headers = {"User-Agent": user_agent}
        self._raw_base = _RAW_BASE_TPL.format(
            owner=self.owner, repo=self.repo, branch=self.branch
        )
        self._api_url = _API_BASE_TPL.format(
            owner=self.owner, repo=self.repo, task=self.task, branch=self.branch
        )
        self.cache_dir: Optional[Path] = (
            Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        )
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = int(max_workers)
        # in-memory cache of available names (populated on first call)
        self._names_cache: Optional[List[str]] = None

    # ---------------------------
    # Representations & OOP niceties
    # ---------------------------
    def __repr__(self) -> str:
        return (
            f"DataLoader(task={self.task!r}, owner={self.owner!r}, repo={self.repo!r}, "
            f"branch={self.branch!r}, cache_dir={str(self.cache_dir) if self.cache_dir else None})"
        )

    def __str__(self) -> str:
        return f"<DataLoader {self.owner}/{self.repo}@{self.branch} task={self.task}>"

    def __len__(self) -> int:
        """Number of available datasets in this task (0 if unknown until fetched)."""
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        """Support ``name in dl`` checks against available names."""
        return name in self.names

    def __iter__(self):
        """Iterate over available dataset base names."""
        yield from self.names

    # ---------------------------
    # Properties
    # ---------------------------
    @property
    def raw_base(self) -> str:
        """Return computed raw base URL for this loader."""
        return self._raw_base

    @property
    def api_url(self) -> str:
        """Return computed GitHub Contents API URL for this task."""
        return self._api_url

    @property
    def names(self) -> List[str]:
        """
        Return the cached list of available dataset base names for the task.

        This property triggers a request to the GitHub Contents API on first access and
        caches the result in memory.

        :return: sorted list of dataset base names (without extension).
        """
        return self.available_names()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _raw_urls_for(self, name: str) -> Dict[str, str]:
        base = f"{self.raw_base}/{self.task}/{name}"
        return {".csv.gz": f"{base}.csv.gz", ".csv": f"{base}.csv"}

    def _cache_path_for(self, name: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return (self.cache_dir / f"{self.task}__{name}.csv.gz").resolve()

    # ---------------------------
    # Public API: names & suggestions
    # ---------------------------
    def available_names(self, refresh: bool = False) -> List[str]:
        """
        Fetch and return a sorted list of available dataset base names for this task using
        the GitHub Contents API.

        The result is cached in memory and refreshed only when ``refresh=True`` is provided.

        :param refresh: Force refresh from GitHub (default: False).
        :return: Sorted list of dataset base names (no extensions).
        """
        if self._names_cache is not None and not refresh:
            return list(self._names_cache)

        try:
            r = requests.get(self.api_url, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            items = r.json()
            names = set()
            for it in items:
                nm = it.get("name", "")
                if nm.endswith(".csv.gz"):
                    names.add(nm[: -len(".csv.gz")])
                elif nm.endswith(".csv"):
                    names.add(nm[: -len(".csv")])
            self._names_cache = sorted(names)
            return list(self._names_cache)
        except requests.RequestException:
            # On failure (rate-limit, network, etc.) return empty cache (but keep it cached)
            self._names_cache = []
            return []

    def refresh_names(self) -> List[str]:
        """
        Force re-fetch available names from GitHub and return them.

        :return: list of available names after refresh.
        """
        return self.available_names(refresh=True)

    def suggest(self, name: str, n: int = 5) -> List[str]:
        """
        Return close matches for ``name`` based on currently available names.

        :param name: requested dataset base name.
        :param n: maximum number of suggestions to return.
        :return: list of suggested names (possibly empty).
        """
        names = self.available_names()
        if not names:
            return []
        return difflib.get_close_matches(name, names, n=n, cutoff=0.4)

    def print_names(self, cols: int = 3, show_count: bool = True) -> None:
        """
        Pretty-print available dataset names in columns.

        :param cols: Number of columns to display (default: 3).
        :param show_count: Whether to show a header with the count (default: True).
        :return: None
        """
        names = self.available_names()
        if show_count:
            print(f"Datasets in task '{self.task}': {len(names)}")
        if not names:
            print("  (no names found or API rate-limited)")
            return
        # column layout
        rows = math.ceil(len(names) / cols)
        # pad to full matrix
        padded = names + [""] * (rows * cols - len(names))
        # fmt: off
        matrix = [padded[i: i + rows] for i in range(0, rows * cols, rows)]
        # fmt: on
        # print rows
        for r in range(rows):
            row_items = [matrix[c][r].ljust(30) for c in range(cols) if matrix[c][r]]
            print("  " + "  ".join(row_items))

    # ---------------------------
    # Main loading functions
    # ---------------------------
    def load(
        self,
        name: str,
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        **pd_kw,
    ) -> pd.DataFrame:
        """
        Load ``Data/<task>/<name>.csv.gz`` (preferred) or ``.csv`` from the repository raw tree.

        :param name: Base filename without extension (e.g. ``"ecoli"``).
        :param use_cache: If True and ``cache_dir`` was provided, use local cached gz bytes when present.
        :param dtype: Optional dtype mapping forwarded to :func:`pandas.read_csv`.
        :param pd_kw: Additional keyword arguments forwarded to :func:`pandas.read_csv`.
        :return: pandas.DataFrame with the loaded dataset.
        :raises FileNotFoundError: If both ``.csv.gz`` and ``.csv`` cannot be fetched (404 or other HTTP error).
        """
        urls = self._raw_urls_for(name)

        # check cache first (only gz cache is stored)
        cache_path = self._cache_path_for(name)
        if use_cache and cache_path is not None and cache_path.exists():
            return pd.read_csv(cache_path, compression="gzip", dtype=dtype, **pd_kw)

        last_err = None
        for ext in [".csv.gz", ".csv"]:
            url = urls[ext]
            try:
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                continue

            if resp.status_code == 200:
                content = resp.content
                # cache gz bytes if applicable
                if use_cache and cache_path is not None and ext == ".csv.gz":
                    try:
                        cache_path.write_bytes(content)
                    except Exception:
                        # ignore caching failures
                        pass

                buf = io.BytesIO(content)
                if ext == ".csv.gz":
                    return pd.read_csv(buf, compression="gzip", dtype=dtype, **pd_kw)
                else:
                    return pd.read_csv(buf, compression=None, dtype=dtype, **pd_kw)
            else:
                last_err = RuntimeError(f"HTTP {resp.status_code} for {url}")

        # both attempts failed — compose helpful message with available names and suggestions
        avail = self.available_names(refresh=True)
        suggestions = self.suggest(name) if avail else []
        tried = [urls[".csv.gz"], urls[".csv"]]
        msg_lines = [
            f"Failed to fetch dataset '{name}' for task '{self.task}'.",
            "Tried URLs (in order):",
        ] + [f"  {u}" for u in tried]

        if avail:
            msg_lines.append("")
            msg_lines.append("Available dataset names (from GitHub API):")
            if len(avail) > 200:
                msg_lines.append(f"  (showing first 200 of {len(avail)}):")
                avail_display = avail[:200]
            else:
                avail_display = avail
            msg_lines += [f"  {n}" for n in avail_display]

            if suggestions:
                msg_lines.append("")
                msg_lines.append(f"Did you mean: {suggestions} ?")

        msg_lines.append("")
        msg_lines.append(f"Last error: {last_err!s}")

        raise FileNotFoundError("\n".join(msg_lines))

    def load_many(
        self,
        names: Iterable[str],
        use_cache: bool = True,
        dtype: Optional[Dict[str, object]] = None,
        parallel: bool = True,
        **pd_kw,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets and return a mapping name -> DataFrame.

        :param names: Iterable of dataset base names to load.
        :param use_cache: If True, use cached gz bytes when present (default True).
        :param dtype: Optional dtype mapping forwarded to pandas.read_csv.
        :param parallel: If True, load in parallel using threads (default True).
        :param pd_kw: Additional kwargs forwarded to pandas.read_csv.
        :return: dict mapping each requested name to its loaded DataFrame.
        :raises RuntimeError: if any single dataset load raises an exception it will be re-raised
                              wrapped in a RuntimeError containing the failing name.
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
