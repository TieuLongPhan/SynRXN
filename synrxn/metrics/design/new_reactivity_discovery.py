"""Metrics for new reactivity discovery."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='new_reactivity_discovery',
    family_id='design',
    default_variant='finite_oracle',
    variants={
        'finite_oracle': metric_variant(
            'observed_optimization',
            primary='budgets.5.scope_hit_group_recovery',
            secondary='budgets.5.source_hit_recovery',
        ),
    },
    qualification=(
        'Score the fraction of hit-bearing substrate groups recovered at the measured budget. Supply both a '
        'source-defined hit threshold and audited candidate group IDs. The secondary score uses the same '
        'threshold and counts individual hits in the measured pool.'
    ),
)
