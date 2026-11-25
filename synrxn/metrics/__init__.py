"""Scientific task metrics and existing chemistry-aware evaluators.

Discovery is lightweight; numeric and chemistry backends load when evaluated.
"""
from .registry import (
    MetricVariant,
    TaskMetricProfile,
    evaluate_task,
    get_task_metrics,
    list_task_metrics,
    metric_protocol_coverage,
)

__all__ = ["MetricVariant", "TaskMetricProfile", "evaluate_task", "get_task_metrics",
           "list_task_metrics", "metric_protocol_coverage"]
