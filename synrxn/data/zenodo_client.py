"""
Thin client for Zenodo lookups, downloads, archive listing/extraction, and name discovery.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import io
import os
import json
import zipfile
import tarfile
import tempfile
import requests
import hashlib

from .constants import ZENODO_RECORD_API, ZENODO_SEARCH_API
from .utils import (
    normalize_version,
    load_json_silent,
    save_json_silent,
    parse_checksum_field,
    sha256_hex,
)

_ARCHIVE_MEMBER_CACHE_PREFIX = "zenodo_archive_members"


class ZenodoClient:
    """
    Zenodo API helper.

    :param session: requests.Session to use for HTTP calls.
    :param cache_dir: Optional Path used for caching record indices, version map, and archive member lists.
    :param cache_record_index: Persist record file indices when True.
    :param timeout: HTTP timeout (seconds).
    """

    def __init__(
        self,
        session: requests.Session,
        cache_dir: Optional[Path] = None,
        cache_record_index: bool = True,
        timeout: int = 20,
    ) -> None:
        self.session = session
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.cache_record_index = bool(cache_record_index)
        self.timeout = int(timeout)

        self._version_map_path = (
            (self.cache_dir / "zenodo_version_map.json") if self.cache_dir else None
        )
        self._version_cache: Dict[str, int] = load_json_silent(self._version_map_path)

    # ---------------- Version & index ----------------
    def resolve_record_id(self, concept_doi: str, version: Optional[str]) -> int:
        """
        Resolve a Zenodo record id for a concept DOI and version (cached when possible).

        :param concept_doi: Concept DOI.
        :param version: Version string or None for latest.
        :return: record id.
        :raises RuntimeError: if no record/version found.
        """
        key = version if version else "__latest__"
        if key in self._version_cache:
            return int(self._version_cache[key])

        params = {"q": f'conceptdoi:"{concept_doi}"', "all_versions": 1, "size": 200}
        r = self.session.get(ZENODO_SEARCH_API, params=params, timeout=self.timeout)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            raise RuntimeError(f"No Zenodo records found for concept DOI {concept_doi}")

        if version:
            target = normalize_version(version)
            for h in hits:
                meta_ver = normalize_version(h.get("metadata", {}).get("version", ""))
                if meta_ver == target:
                    rid = int(h["id"])
                    self._version_cache[key] = rid
                    save_json_silent(self._version_map_path, self._version_cache)
                    return rid
            for h in hits:
                raw = str(h.get("metadata", {}).get("version", "")).strip()
                if raw == version or raw == f"v{version}" or f"v{raw}" == version:
                    rid = int(h["id"])
                    self._version_cache[key] = rid
                    save_json_silent(self._version_map_path, self._version_cache)
                    return rid
            raise RuntimeError(
                f"Version '{version}' not found under {concept_doi}. "
                f"Available: {sorted({h.get('metadata', {}).get('version','') for h in hits})}"
            )

        hits_sorted = sorted(
            hits, key=lambda h: h.get("updated", h.get("created", "")), reverse=True
        )
        rid = int(hits_sorted[0]["id"])
        self._version_cache[key] = rid
        save_json_silent(self._version_map_path, self._version_cache)
        return rid

    def build_file_index(self, record_id: int) -> Dict[str, Dict]:
        """
        Build and return file index mapping 'key' -> metadata for a Zenodo record.

        :param record_id: Zenodo record id.
        :return: dict of key->meta
        """
        if self.cache_dir and self.cache_record_index:
            rp = self.cache_dir / f"zenodo_record_{record_id}.json"
            cached = load_json_silent(rp)
            if cached:
                if isinstance(cached, dict):
                    return cached
                if isinstance(cached, list):
                    return {f.get("key", ""): f for f in cached if f.get("key")}

        url = ZENODO_RECORD_API.format(record_id=record_id)
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        meta = r.json()
        files = meta.get("files", [])
        if isinstance(files, list):
            index = {f.get("key", ""): f for f in files if f.get("key")}
        else:
            index = files

        if self.cache_dir and self.cache_record_index:
            rp = self.cache_dir / f"zenodo_record_{record_id}.json"
            save_json_silent(rp, index)
        return index

    # ---------------- Search & names ----------------
    @staticmethod
    def find_keys(file_index: Dict[str, Dict], term: str) -> List[str]:
        """
        Case-insensitive search over a Zenodo file index dict.

        :param file_index: Mapping key->meta.
        :param term: Substring to match (lowercased).
        :return: Matching keys.
        """
        t = term.lower()
        return [k for k in file_index.keys() if t in k.lower()]

    def _archive_cache_path(self, record_id: int, archive_key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = sha256_hex(f"{record_id}::{archive_key}")[:16]
        fname = f"{_ARCHIVE_MEMBER_CACHE_PREFIX}__{record_id}__{h}.json"
        return (self.cache_dir / fname).resolve()

    def list_archive_members_cached(
        self,
        record_id: int,
        archive_key: str,
        meta: Dict,
    ) -> List[str]:
        """
        Return the list of member paths inside an attached archive; cache listing to disk.
        Downloads the archive only if no cache is present.

        :param record_id: Zenodo record id.
        :param archive_key: Zenodo file 'key' for the archive.
        :param meta: Zenodo file meta dict.
        :return: list of member paths (strings).
        """
        cp = self._archive_cache_path(record_id, archive_key)
        if cp and cp.exists():
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                pass

        resp = self.get_download_response(meta, record_id)
        if resp is None:
            return []

        temp_path = None
        members: List[str] = []
        try:
            suffix = ".zip" if archive_key.lower().endswith(".zip") else ".tar"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                temp_path = Path(tf.name)
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tf.write(chunk)

            if archive_key.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(temp_path) as z:
                        members = z.namelist()
                except Exception:
                    members = []
            else:
                try:
                    with tarfile.open(temp_path, mode="r:*") as tar:
                        members = [m.name for m in tar.getmembers()]
                except Exception:
                    members = []
        finally:
            try:
                resp.close()
            except Exception:
                pass
            try:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

        try:
            if cp:
                cp.write_text(json.dumps(members), encoding="utf-8")
        except Exception:
            pass

        return members

    def available_names(
        self,
        task: str,
        record_id: int,
        file_index: Dict[str, Dict],
        include_archives: bool = True,
    ) -> List[str]:
        """
        Discover dataset names under Data/{task}/, including archive members (if requested).

        :param task: Task subfolder name.
        :param record_id: Zenodo record id.
        :param file_index: Mapping of key->meta for the record.
        :param include_archives: If True, inspect archives to list inner members under Data/{task}/.
        :return: Sorted list of base names.
        """
        base = f"Data/{task}/"
        names = set()

        # direct files
        for key in file_index.keys():
            if not key.startswith(base):
                continue
            if key.endswith(".csv.gz"):
                names.add(key[len(base) : -len(".csv.gz")])
            elif key.endswith(".csv"):
                names.add(key[len(base) : -len(".csv")])

        if not include_archives:
            return sorted(names)

        # archives
        archive_keys = [
            k
            for k in file_index.keys()
            if k.lower().endswith((".zip", ".tar.gz", ".tgz", ".tar"))
        ]
        needle = f"data/{task}/"  # search anywhere in members
        for ak in archive_keys:
            try:
                meta = file_index.get(ak, {})
                members = self.list_archive_members_cached(record_id, ak, meta)
                if not members:
                    continue
                for m in members:
                    ml = m.replace("\\", "/")
                    ml_low = ml.lower()
                    idx = ml_low.find("/" + needle)
                    if idx == -1:
                        idx = ml_low.find(needle)
                        if idx > 0 and ml_low[idx - 1] != "/":
                            continue
                        if idx == -1:
                            continue
                    tail = ml[idx + len(needle) :]
                    if tail.endswith(".csv.gz"):
                        names.add(tail[: -len(".csv.gz")])
                    elif tail.endswith(".csv"):
                        names.add(tail[: -len(".csv")])
            except Exception:
                continue

        return sorted(names)

    # ---------------- Download & extraction ----------------
    def get_download_response(
        self, meta: Dict, record_id: int
    ) -> Optional[requests.Response]:
        """
        Try candidate URLs for a given file metadata and return first streaming Response.

        :param meta: Zenodo file metadata dict (contains 'links' and 'key').
        :param record_id: Zenodo record id (used to construct canonical URLs).
        :return: requests.Response (streaming) or None.
        """
        links = (meta.get("links") or {}) or {}
        candidates: List[str] = []

        dl = links.get("download")
        if dl:
            candidates.append(dl)

        self_link = links.get("self")
        if self_link:
            candidates.append(self_link)
            candidates.append(
                self_link + ("&download=1" if "?" in self_link else "?download=1")
            )

        key = meta.get("key", "")
        if key and record_id is not None:
            filename = key.split("/")[-1]
            candidates.append(
                f"https://zenodo.org/record/{record_id}/files/{filename}?download=1"
            )
            candidates.append(
                f"https://zenodo.org/record/{record_id}/files/{key}?download=1"
            )

        html = links.get("html")
        if html:
            candidates.append(html)

        seen = set()
        uniq: List[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)

        for url in uniq:
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=True)
            except Exception:
                continue
            ct = (resp.headers.get("Content-Type") or "").lower()
            if resp.status_code == 200 and "application/json" not in ct:
                return resp
            try:
                resp.close()
            except Exception:
                pass
        return None

    def stream_to_temp_and_verify(
        self, resp: requests.Response, meta: Dict, suffix: str
    ) -> Path:
        """
        Stream a response to a temporary file and verify checksum if provided.

        :param resp: Streaming HTTP response.
        :param meta: Zenodo file metadata (may contain 'checksum').
        :param suffix: Suggested file suffix (e.g., '.zip' or '.csv').
        :return: Path to the temp file (caller is responsible for deleting it).
        :raises RuntimeError: if checksum mismatch occurs.
        """
        algo, expected_hex = parse_checksum_field(meta.get("checksum", ""))
        h = hashlib.new(algo) if (algo and expected_hex) else None

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            temp_path = Path(tf.name)
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                tf.write(chunk)
                if h:
                    h.update(chunk)

        if h and h.hexdigest().lower() != str(expected_hex).lower():
            raise RuntimeError("Checksum mismatch")

        return temp_path

    def extract_member_bytes(
        self,
        archive_path: Path,
        member_name: str,
    ) -> Optional[bytes]:
        """
        Extract one member file from a .zip or .tar* archive into memory.

        :param archive_path: Path to the downloaded archive.
        :param member_name: Member path to extract (as appears in the archive).
        :return: bytes or None if not found.
        """
        p = str(archive_path).lower()
        if p.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as z:
                try:
                    with z.open(member_name) as mf:
                        return mf.read()
                except KeyError:
                    return None
        else:
            with tarfile.open(archive_path, mode="r:*") as tar:
                try:
                    fobj = tar.extractfile(member_name)
                    if fobj:
                        return fobj.read()
                except KeyError:
                    return None
        return None
