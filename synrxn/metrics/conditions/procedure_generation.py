"""Metrics for procedure generation."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='procedure_generation',
    family_id='conditions',
    default_variant='actions',
    variants={
        'actions': metric_variant(
            'sequence_recovery',
            primary='mean_normalized_edit_distance',
            secondary='exact_match',
        ),
    },
    qualification=(
        'Reference action agreement does not establish procedure safety or successful laboratory execution; '
        'withhold procedures from model inputs.'
    ),
)
