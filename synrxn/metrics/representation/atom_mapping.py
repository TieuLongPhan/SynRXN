"""Metrics for atom mapping."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='atom_mapping',
    family_id='representation',
    default_variant='reference_contract',
    variants={
        'reference_contract': metric_variant('reference_contract', primary='top1_accuracy', secondary='topk_accuracy'),
    },
    qualification=(
        'Requires complete or product-origin correspondence scope, reference tier and symmetry policy in '
        'the validated contract; no bare integer-array equality.'
    ),
)
