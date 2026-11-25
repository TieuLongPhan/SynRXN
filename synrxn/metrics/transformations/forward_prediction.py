"""Metrics for forward prediction."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='forward_prediction',
    family_id='transformations',
    default_variant='reference_contract',
    variants={
        'reference_contract': metric_variant('reference_contract', primary='top1_accuracy', secondary='topk_accuracy'),
    },
    qualification=(
        'Product-isolated instances and accepted product multisets are scored by the chemical reference '
        'contract, retaining isotope, charge, stereochemistry and multiplicity.'
    ),
)
