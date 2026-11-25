"""Metrics for mechanistic prediction."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='mechanistic_prediction',
    family_id='mechanisms',
    default_variant='strict_replay',
    variants={
        'strict_replay': metric_variant(
            'mechanism',
            primary='top_k.5.trajectory_recovery',
            secondary='top_k.5.event_recovery',
        ),
    },
    qualification=(
        'Endpoints are supplied; endpoint agreement is diagnostic. Event/trajectory recovery requires '
        'strict replay validity and source/reference tiers.'
    ),
)
