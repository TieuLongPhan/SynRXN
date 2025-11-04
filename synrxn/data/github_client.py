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
    :param ref_candidates: Ordered list of (ref_type, ref) pairs.
                          The `ref` can be a branch, a tag, or a commit SHA.
                          The `ref_type` is informational and not required for URLs.
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
        List base filenames under Data/{task}/ from the first successful ref.

        :param task: Task subfolder name.
        :return: Sorted list of names without extension.
        """
        names = set()
        for _, ref in self.ref_candidates:
            api_url = GH_API_TPL.format(
                owner=self.owner, repo=self.repo, task=task, ref=ref
            )
            try:
                r = self.session.get(api_url, timeout=self.timeout)
                r.raise_for_status()
                items = r.json()
            except requests.RequestException:
                continue
            if not isinstance(items, list):
                continue
            for it in items:
                nm = it.get("name", "")
                if nm.endswith(".csv.gz"):
                    names.add(nm[: -len(".csv.gz")])
                elif nm.endswith(".csv"):
                    names.add(nm[: -len(".csv")])
            break  # only first working ref
        return sorted(names)

    def raw_url(self, task: str, name: str, ext: str) -> Optional[str]:
        """
        Return a working raw.githubusercontent URL for the file candidate, or None.

        :param task: Task subfolder.
        :param name: Base file name.
        :param ext: Extension ('.csv' or '.csv.gz').
        :return: URL string or None.
        """
        for _, ref in self.ref_candidates:
            base = GH_RAW_TPL.format(owner=self.owner, repo=self.repo, ref=ref)
            url = f"{base}/{task}/{name}{ext}"
            try:
                r = self.session.get(url, timeout=self.timeout, stream=True)
                if r.status_code == 200:
                    r.close()
                    return url
            except Exception:
                pass
        return None

    def latest_commit_sha(self, branch: Optional[str] = None) -> Optional[str]:
        """
        Resolve the latest commit SHA for `branch`. If `branch` is None, query
        the repository to learn its default branch first.

        Returns the 40-char commit SHA or None on failure.
        """
        # find default branch if not provided
        if branch is None:
            repo_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
            try:
                r = self.session.get(repo_url, timeout=self.timeout)
                r.raise_for_status()
                branch = r.json().get("default_branch") or "main"
            except Exception:
                branch = "main"

        # Query the commits endpoint for the branch (this returns the latest commit for that ref)
        commits_url = (
            f"https://api.github.com/repos/{self.owner}/{self.repo}/commits/{branch}"
        )
        try:
            r = self.session.get(commits_url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            sha = data.get("sha")
            if sha and isinstance(sha, str):
                return sha
        except Exception:
            pass
        return None
