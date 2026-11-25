"""Task split requirements and deterministic dependence-aware strategies.

Existing repeated-k-fold classes remain available from their original modules.
The task protocol layer has no NumPy/pandas/scikit-learn import requirement.
"""

from .protocols import (
    TaskSplitProtocol,
    get_task_split_protocol,
    get_task_split_protocols,
    split_protocol_coverage,
)
from .strategies import STRATEGIES, make_task_split

__all__ = [
    "TaskSplitProtocol", "get_task_split_protocol", "get_task_split_protocols",
    "split_protocol_coverage", "STRATEGIES", "make_task_split",
]
