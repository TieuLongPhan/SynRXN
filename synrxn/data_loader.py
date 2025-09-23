"""
dataloader.py
--------------------

An object-oriented DataLoader for downloading dataset files from a Hugging Face
dataset repository (e.g. "TieuLongPhan/rxndb").

Notes
-----
* The docstrings use Sphinx/reST field lists (``:param:``, ``:type:``, ``:returns:``,
  ``:rtype:``, ``:raises:``) so they work well with Sphinx (autodoc) + Napoleon.
* Authentication: set environment variable ``HUGGINGFACE_TOKEN`` for private repos,
  or run ``huggingface-cli login``.
* Requirements: ``huggingface_hub`` (install with `pip install huggingface_hub`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download


__all__ = ["DataLoader"]


class DataLoader:
    """
    Object-oriented DataLoader for Hugging Face dataset repositories.

    The typical usage pattern is to instantiate the loader once (configure repo,
    prefix, extension, cache directory) and then call :meth:`download` with only
    a single `data_name` parameter (e.g., "SYNTEMP", "synTemp", "syntemp").

    :param repo_id: Hugging Face dataset repo id (user/repo). Default: "TieuLongPhan/rxndb".
    :type repo_id: str
    :param prefix: Folder inside the dataset repo containing files. Trailing slash
        is optional. Default: "Data/".
    :type prefix: str
    :param ext: File extension appended to normalized names. Default: ".csv.gz".
    :type ext: str
    :param out_dir: Local directory to copy downloaded files to. If None, uses
        "~/.cache/synrxn/<repo_id>/". Default: None.
    :type out_dir: Path | str | None
    :param revision: Repo revision (branch, tag, commit). Default: "main".
    :type revision: str
    :param use_manifest: If True, attempt to load "<prefix>/manifest.json" for
        SHA-256 verification. Default: True.
    :type use_manifest: bool
    :param alias_map: Optional mapping of alias -> canonical names. Keys/values
        will be normalized. Example: {"syn-temp": "syntemp"}.
    :type alias_map: dict[str, str] | None

    Example
    -------
    Basic usage (only change `data_name`):

    .. code-block:: python

        from rxndb_dataloader import DataLoader

        dl = DataLoader()
        local_path = dl.download("SYNTEMP")
        print(local_path)

    :notes: If the repository is private, set HUGGINGFACE_TOKEN in the environment.
    """

    def __init__(
        self,
        repo_id: str = "TieuLongPhan/rxndb",
        prefix: str = "Data/",
        ext: str = ".csv.gz",
        out_dir: Optional[Path | str] = None,
        revision: str = "main",
        use_manifest: bool = True,
        alias_map: Optional[Dict[str, str]] = None,
    ) -> None:
        # repo + path configuration
        self.repo_id = repo_id
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.ext = ext if ext.startswith(".") else f".{ext}"
        self.revision = revision

        # HF API helper
        self.api = HfApi()

        # output/caching directory (deterministic)
        self.out_dir = (
            Path(out_dir)
            if out_dir
            else Path.home() / ".cache" / "synrxn" / repo_id.replace("/", "_")
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # manifest & alias handling
        self.use_manifest = bool(use_manifest)
        self._manifest: Optional[Dict[str, Any]] = None
        self.alias_map = {
            self._normalize(k): self._normalize(v) for k, v in (alias_map or {}).items()
        }

        # logging
        self.log = logging.getLogger(self.__class__.__name__)
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.log.addHandler(handler)
        self.log.setLevel(logging.INFO)

        # try to load manifest if requested
        if self.use_manifest:
            self._try_load_manifest()

    def download(self, data_name: str, overwrite: bool = False) -> Path:
        """
        Download a single dataset file by name and return its local path.

        :param data_name: Data handle to download (e.g. "SynTemp", "syntemp", "SYNTEMP").
                          The name will be normalized (lowercase, alphanumeric).
        :type data_name: str
        :param overwrite: If True, replace local file even if it exists and passes checksum.
        :type overwrite: bool
        :returns: Local path to the downloaded file (inside ``out_dir``).
        :rtype: pathlib.Path
        :raises FileNotFoundError: If the resolved remote file does not exist.
        :raises IOError: If checksum verification fails when a manifest is available.
        """
        canonical = self._canonical_name(data_name)
        remote_path = f"{self.prefix}{canonical}{self.ext}"

        if not self._exists_remote(remote_path):
            raise FileNotFoundError(
                f"Remote file not found: {remote_path} in {self.repo_id} (rev={self.revision})"
            )

        self.log.info(
            "Downloading %s from %s (rev=%s)", remote_path, self.repo_id, self.revision
        )
        # this returns a path inside HF cache
        cached = hf_hub_download(
            repo_id=self.repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=self.revision,
        )

        dest = self.out_dir / f"{canonical}{self.ext}"

        # if already exists and not overwriting, verify (if manifest provided) or return
        if dest.exists() and not overwrite:
            expected = self._expected_sha(f"{canonical}{self.ext}")
            if expected:
                actual = self._sha256(dest)
                if actual.lower() == expected.lower():
                    self.log.info("Local file exists and checksum OK: %s", dest)
                    return dest
                self.log.warning(
                    "Local checksum mismatch; will overwrite from cache: %s", dest
                )
            else:
                self.log.info(
                    "Local file exists; returning without checksum verification: %s",
                    dest,
                )
                return dest

        # copy from HF cache to deterministic location
        shutil.copyfile(cached, dest)
        self.log.info("Copied to %s", dest)

        # verify checksum if manifest is present
        expected = self._expected_sha(f"{canonical}{self.ext}")
        if expected:
            actual = self._sha256(dest)
            if actual.lower() != expected.lower():
                dest.unlink(missing_ok=True)
                raise IOError(
                    f"Checksum mismatch for {dest.name}: expected {expected}, got {actual}"
                )
            self.log.info("Checksum OK for %s", dest.name)

        return dest

    def available(self) -> List[str]:
        """
        Return a sorted list of available (normalized) dataset names under the
        configured prefix and extension.

        :returns: Sorted list of canonical dataset names.
        :rtype: list[str]
        """
        files = self._list_remote_files()
        names = []
        for f in files:
            if not f.startswith(self.prefix):
                continue
            base = Path(f).name
            if base.endswith(self.ext):
                name = base[: -len(self.ext)]
                names.append(self._normalize(name))
        return sorted(set(names))

    # ---------------------------- Internal helpers ----------------------------

    def _canonical_name(self, raw: str) -> str:
        """Normalize and apply alias map to get canonical dataset name."""
        n = self._normalize(raw)
        return self.alias_map.get(n, n)

    @staticmethod
    def _normalize(s: str) -> str:
        """
        Normalize dataset name to a canonical lowercase alphanumeric string.

        Examples
        --------
        >>> DataLoader._normalize("SynTemp")
        'syntemp'
        >>> DataLoader._normalize("syn temp!!")
        'syntemp'

        :param s: input string to normalize
        :type s: str
        :returns: normalized string
        :rtype: str
        """
        s = (s or "").lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    def _try_load_manifest(self) -> None:
        """
        Attempt to download and parse '<prefix>manifest.json' from the repo.
        If unavailable, the loader will continue without checksums.

        :returns: None
        """
        manifest_remote = f"{self.prefix}manifest.json"
        try:
            local_manifest = hf_hub_download(
                repo_id=self.repo_id,
                filename=manifest_remote,
                repo_type="dataset",
                revision=self.revision,
            )
            with open(local_manifest, "r", encoding="utf-8") as fh:
                self._manifest = json.load(fh)
            self.log.info(
                "Loaded manifest.json (entries: %d)",
                len(self._manifest.get("files", [])),
            )
        except Exception:
            # manifest is optional — continue without it
            self._manifest = None
            self.log.debug(
                "No manifest.json found at %s (rev=%s)", manifest_remote, self.revision
            )

    def _expected_sha(self, filename: str) -> Optional[str]:
        """
        Return expected sha256 for a filename (basename or prefixed path) from manifest.

        :param filename: filename (basename or prefixed) to look up in manifest
        :type filename: str
        :returns: hex sha256 digest or None if not found
        :rtype: str | None
        """
        if not self._manifest:
            return None
        files = self._manifest.get("files", []) or []
        for entry in files:
            path = entry.get("path") or entry.get("filename") or entry.get("name")
            if not path:
                continue
            if (
                path == filename
                or path == f"{self.prefix}{filename}"
                or Path(path).name == Path(filename).name
            ):
                return (entry.get("sha256") or entry.get("hash") or "").strip() or None
        return None

    def _exists_remote(self, path_in_repo: str) -> bool:
        """
        Check remote existence by listing repo files and testing membership.

        :param path_in_repo: full path inside repo (including prefix)
        :type path_in_repo: str
        :returns: True if path exists remotely
        :rtype: bool
        """
        try:
            files = self._list_remote_files()
            return path_in_repo in files
        except Exception:
            return False

    def _list_remote_files(self) -> List[str]:
        """
        List files in the dataset repo at the configured revision.

        :returns: list of file paths in the repo
        :rtype: list[str]
        """
        return self.api.list_repo_files(
            repo_id=self.repo_id, repo_type="dataset", revision=self.revision
        )

    @staticmethod
    def _sha256(p: Path, chunk_size: int = 1 << 20) -> str:
        """
        Compute SHA-256 of a local file.

        :param p: path to file
        :type p: pathlib.Path
        :param chunk_size: read chunk size in bytes
        :type chunk_size: int
        :returns: hex digest string
        :rtype: str
        """
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
