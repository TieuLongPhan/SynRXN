"""Metrics for mutation effects."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='mutation_effects',
    family_id='biocatalysis',
    default_variant='native_delta',
    variants={
        'native_delta': metric_variant('regression', primary='mae', secondary='direction_accuracy', signed=True),
    },
    qualification=(
        'Score within assay/campaign native deltas and resolved direction conventions. Do not pool assay '
        'units or treat all deep-mutation proxies as reaction-specific activity.'
    ),
)
