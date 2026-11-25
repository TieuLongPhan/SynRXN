"""Metrics for reaction clustering."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_clustering',
    family_id='representation',
    default_variant='external_labels',
    variants={
        'external_labels': metric_variant(
            'clustering',
            primary='adjusted_rand_index',
            secondary='normalized_mutual_information',
        ),
    },
    qualification=(
        'Do not expose reference labels to clustering or select k from test labels. Stability is a separate '
        'repeated-run analysis.'
    ),
)
