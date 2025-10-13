"""
Thin client for GitHub listing and raw-file fetch.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import requests
from .constants import GH_API_TPL, GH_RAW_TPL


class GitHubClient:
    """
    GitHub helper for listing and raw content retrieval.

    :param session: requests.Session configured by caller.
    :param timeout: HTTP timeout (seconds).
    :param owner: GitHub repository owner.
    :param repo: GitHub repository name.
    :param ref_candidates: Ordered list of (ref_type, ref) pairs (e.g., ('tags','v0.0.5')).
    """

    def __init__(
        self,
        session: requests.Session,
        timeout: int,
        owner: str,
        repo: str,
        ref_candidates: Iterable[Tuple[str, str]],
    ) -> None:
        self.session = session
        self.timeout = int(timeout)
        self.owner = owner
        self.repo = repo
        self.ref_candidates = list(ref_candidates)

    def list_names(self, task: str) -> List[str]:
        """
        List base filenames under Data/{task}/ from first successful ref.

        :param task: Task subfolder name.
        :return: Sorted list of names without extension.
        """
        names = set()
        for ref_type, ref in self.ref_candidates:
            api_url = GH_API_TPL.format(
                owner=self.owner, repo=self.repo, task=task, ref=ref
            )
            try:
                r = self.session.get(api_url, timeout=self.timeout)
                r.raise_for_status()
                items = r.json()
            except requests.RequestException:
                continue
            for it in items:
                nm = it.get("name", "")
                if nm.endswith(".csv.gz"):
                    names.add(nm[: -len(".csv.gz")])
                elif nm.endswith(".csv"):
                    names.add(nm[: -len(".csv")])
            break
        return sorted(names)

    def raw_url(self, task: str, name: str, ext: str) -> Optional[str]:
        """
        Return a working raw.githubusercontent URL for the file candidate, or None.

        :param task: Task subfolder.
        :param name: Base file name.
        :param ext: Extension ('.csv' or '.csv.gz').
        :return: URL string or None.
        """
        for ref_type, ref in self.ref_candidates:
            base = GH_RAW_TPL.format(
                owner=self.owner, repo=self.repo, ref_type=ref_type, ref=ref
            )
            url = f"{base}/{task}/{name}{ext}"
            try:
                r = self.session.get(url, timeout=self.timeout, stream=True)
                if r.status_code == 200:
                    r.close()
                    return url
            except Exception:
                pass
        return None
