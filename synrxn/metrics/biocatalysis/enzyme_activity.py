"""Metrics for enzyme activity."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='enzyme_activity',
    family_id='biocatalysis',
    default_variant='native_labels',
    variants={
        'native_labels': metric_variant('classification', primary='macro_f1', secondary='balanced_accuracy'),
        'positive_probability': metric_variant('binary', primary='average_precision', secondary='brier_score'),
    },
    qualification=(
        'Preserve native assay setting and activity definition; no general transfer to unseen enzymes or '
        'reaction conditions is inferred.'
    ),
)
