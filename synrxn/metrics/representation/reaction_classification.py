"""Metrics for reaction classification."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_classification',
    family_id='representation',
    default_variant='labels',
    variants={
        'labels': metric_variant('classification', primary='macro_f1', secondary='accuracy'),
    },
    qualification=(
        'Native taxonomies remain separate; hierarchical scores require a declared source taxonomy and are '
        'not implemented by the label scorer.'
    ),
)
