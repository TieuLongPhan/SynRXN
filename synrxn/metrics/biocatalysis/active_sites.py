"""Metrics for active sites."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='active_sites',
    family_id='biocatalysis',
    default_variant='observed_positive_ranking',
    variants={
        'observed_positive_ranking': metric_variant(
            'ranked_retrieval',
            primary='top_k.5.recall',
            secondary='top_k.5.hit_rate',
            observed_only=True,
        ),
        'residue_probability': metric_variant('residue_probabilities', primary='protein_macro_average_precision'),
        'residue_sets': metric_variant('set_recovery', primary='macro_f1', secondary='exact_match'),
    },
    qualification=(
        'Use native residue numbering and protein grouping. Current M-CSA zeros are unannotated rather than '
        'proven noncatalytic; default ranking recovers supplied positive residue IDs. Probability/F1 '
        'metrics require separately qualified complete labels or an explicitly declared annotation-recovery '
        'proxy.'
    ),
)
