"""Metrics for literature extraction."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='literature_extraction',
    family_id='data',
    default_variant='entities',
    variants={
        'entities': metric_variant('set_recovery', primary='micro_f1', secondary='macro_f1'),
        'relations': metric_variant('set_recovery', primary='micro_f1', secondary='macro_f1'),
    },
    qualification=(
        'Score entity spans/types and relation tuples separately per document; annotation exhaustiveness '
        'and entity normalization must be qualified.'
    ),
)
