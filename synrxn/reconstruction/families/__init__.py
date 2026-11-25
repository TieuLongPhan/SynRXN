"""Reconstruct source tables by scientific family or task."""

from ..paths import REPO
from ._selection import plan, read_registry, run

__all__ = ["definitions", "plan", "run"]


def definitions(repo=REPO):
    """Load scientific family definitions from the task registry."""
    return read_registry(repo)["families"]
