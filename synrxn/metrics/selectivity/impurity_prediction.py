"""Metrics for impurity prediction."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='impurity_prediction',
    family_id='selectivity',
    default_variant='observed_identities',
    variants={
        'observed_identities': metric_variant(
            'set_recovery',
            primary='observed_recall',
            secondary='query_macro_recall',
            observed_only=True,
        ),
        'complete_identities': metric_variant('set_recovery', primary='exact_match', secondary='macro_f1'),
        'native_yield': metric_variant('regression', primary='mae', secondary='rmse'),
        'ranked_formulas': metric_variant(
            'ranked_retrieval',
            primary='top_k.5.recall',
            secondary='top_k.5.reciprocal_rank',
        ),
    },
    qualification=(
        'Unreported impurities are unknown. Precision/exact match require a qualified complete-native- '
        'identity subset; numeric benzaldehyde yield is a different endpoint.'
    ),
)
