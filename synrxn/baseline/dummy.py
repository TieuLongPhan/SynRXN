"""Train-only weak controls and explicit limits for scientific task variants.

These predictors never accept held-out references. Oracle copies, deliberately
wrong outputs and reference permutations belong in evaluator tests instead.
Discovery imports neither numerical nor chemical backends.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import random
from statistics import mean, median
from typing import Any

from synrxn.metrics.registry import get_task_metrics, list_task_metrics


@dataclass(frozen=True)
class DummyProtocol:
    """Describe supported controls without implying source readiness.

    :param task_id: Scientific task identifier.
    :param variant: Metric variant within the task.
    :param family_id: Scientific family identifier.
    :param scorer: Required evaluator input schema.
    :param strategies: Implemented weak prediction strategies.
    :param status: Generic control or specialized adapter requirement.
    :param requirements: Additional source-specific input requirements.
    :param qualification: Limits on scientific interpretation.
    """

    task_id: str
    variant: str
    family_id: str
    scorer: str
    strategies: tuple[str, ...]
    status: str
    requirements: tuple[str, ...]
    qualification: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable profile snapshot.

        :return: Independent profile fields.
        """
        return asdict(self)


_STRATEGIES = {
    "regression": ("train_mean", "train_median", "train_sample"),
    "classification": ("train_majority", "train_prior", "uniform_labels"),
    "binary": ("train_prior", "uniform_probability", "random_probability"),
    "composition": ("train_mean", "train_median", "train_sample"),
    "set_recovery": ("empty", "train_sample"),
    "sequence_recovery": ("empty", "train_sample"),
    "ranked_retrieval": ("random_order",),
    "observed_optimization": ("random_order",),
    "clustering": ("one_cluster", "random_clusters"),
    "trajectory": ("causal_persistence",),
    "residue_probabilities": ("train_prior", "uniform_probability", "random_probability"),
}
_SPECIALIZED = {
    "reference_contract": "Chemical candidates must satisfy the typed correspondence/product contract; use a specialized adapter, such as the existing forward retrieval control.",
    "mechanism": "Mechanism candidates require typed records, endpoint policies and strict chemical replay; random events are not a qualified mechanism baseline.",
    "geometry": "Geometry controls require atom correspondence, coordinate units and a declared physical template; arbitrary coordinates are not a qualified TS baseline.",
    "control": "A controller must generate actual held-out plant/simulator rollouts; copying desired traces or randomizing recorded states is not a controller baseline.",
    "energy_forces": "Energy/force controls require configuration, atom-count and native-unit adapters; independent random numbers do not define a consistent reactive potential.",
}


def get_dummy_protocol(task_id: str, variant: str | None = None) -> DummyProtocol:
    """Discover controls for one task variant without loading source data.

    :param task_id: Registered scientific task identifier.
    :param variant: Metric variant, or the task default when omitted.
    :return: Immutable prediction requirements and available strategies.
    :raises KeyError: If the task or variant is unknown.
    """
    task = get_task_metrics(task_id)
    selected = task.default_variant if variant is None else variant
    if selected not in task.variants:
        raise KeyError(f"unknown dummy variant {task_id}/{selected}")
    scorer = task.variants[selected].scorer
    strategies = _STRATEGIES.get(scorer, ())
    if task_id == "reaction_progress_monitoring" and scorer == "regression":
        strategies += ("causal_persistence",)
    if task_id == "mutation_effects" and scorer == "regression":
        strategies += ("zero_change",)
    requirements = ["Fit target statistics and label frequencies using training records only."]
    if scorer in {"ranked_retrieval", "observed_optimization"}:
        requirements.append("Supply available candidate IDs without relevance labels or oracle response values; order is sampled without replacement.")
        requirements.append("Candidate availability is a separate source contract; unreported associations are unknown, not confirmed negatives.")
    if "causal_persistence" in strategies:
        requirements.append("Supply only the last permitted observed state, plus requested horizon lengths for trajectory outputs; no future targets.")
    if scorer == "residue_probabilities":
        requirements.append("Supply eligible residue counts and complete training residue labels; retain protein grouping in evaluation.")
    if scorer == "composition":
        requirements.append("Pin component order and native scale upstream; no normalization or missing-mass imputation is performed.")
    if "zero_change" in strategies:
        requirements.append("The zero-change control predicts a native mutation delta of zero; no training targets or wild-type test measurements are needed.")
    return DummyProtocol(
        task_id, selected, task.family_id, scorer, strategies,
        "generic_control" if strategies else "specialized_adapter_required",
        tuple(requirements),
        _SPECIALIZED.get(scorer, "A weak statistical control, not a chemical validity guarantee or evidence that a table is benchmark-ready.")
        + " " + task.qualification,
    )


def list_dummy_protocols(family_id: str | None = None) -> dict[str, dict[str, DummyProtocol]]:
    """Discover every scientific task and all its metric variants.

    :param family_id: Optional registered scientific family filter.
    :return: Task identifiers mapped to variant-specific profiles.
    """
    return {task_id: {variant: get_dummy_protocol(task_id, variant)
                     for variant in task.variants}
            for task_id, task in list_task_metrics(family_id).items()}


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("numeric training references and permitted inputs must be finite")
    return float(value)


def _label(value: Any) -> Any:
    if value is None or isinstance(value, Real) and not math.isfinite(value):
        raise ValueError("labels must be finite, nonmissing and hashable")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("labels must be finite, nonmissing and hashable") from exc
    return value


def _identifiers(values: Sequence[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a sequence of identifiers")
    result = tuple(values)
    if not result or any(type(value) not in (str, int)
                         or isinstance(value, str) and not value.strip() for value in result):
        raise ValueError(f"{name} must contain nonempty string or integer identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


class DummyPredictor:
    """Generate weak predictions from training statistics or allowed inputs.

    :param task_id: Registered scientific task identifier.
    :param strategy: One strategy returned by :func:`get_dummy_protocol`.
    :param variant: Optional metric variant.
    :param seed: Integer random seed, independent of Python's hash seed.
    :param n_clusters: Predeclared cluster count for random clustering only.
    :raises ValueError: If the strategy requires a specialized adapter.

    ``predict`` has no reference/target argument. Its outputs follow the
    selected scorer's array schema and can be passed to ``evaluate_task``.
    Randomness depends on the instance ID, so batching and input order do not
    change an instance's prediction. Callers must still audit source lineage
    and certify that ``fit`` receives training-only targets.
    """

    def __init__(self, task_id: str, strategy: str, *, variant: str | None = None,
                 seed: int = 0, n_clusters: int = 2):
        self.protocol = get_dummy_protocol(task_id, variant)
        if strategy not in self.protocol.strategies:
            raise ValueError(f"unsupported dummy strategy {strategy!r}: {self.protocol.qualification}")
        if type(seed) is not int or type(n_clusters) is not int or n_clusters < 1:
            raise ValueError("seed must be an integer and n_clusters a positive integer")
        self.strategy, self.seed, self.n_clusters = strategy, seed, n_clusters
        self._fitted = False
        self._training_ids: frozenset[Any] = frozenset()
        self._targets: list[Any] = []
        self._labels: list[Any] = []
        self._counts: list[int] = []
        self._constant: Any = None

    def fit(self, training_targets: Sequence[Any], *, training_ids: Sequence[Any],
            label_universe: Sequence[Any] | None = None) -> DummyPredictor:
        """Fit statistics from an explicitly identified training partition.

        :param training_targets: Qualified training targets in native units.
        :param training_ids: Unique IDs aligned to the training targets.
        :param label_universe: Optional declared classification label universe.
        :return: This fitted predictor.
        :raises ValueError: If targets are empty, nonfinite or malformed.
        """
        self._fitted = False
        self._targets, self._labels, self._counts = [], [], []
        self._constant = None
        self._training_ids = frozenset()
        ids = _identifiers(training_ids, "training_ids")
        targets = deepcopy(list(training_targets))
        if len(targets) != len(ids):
            raise ValueError("training references and training_ids must have equal nonzero length")
        scorer = self.protocol.scorer
        if scorer == "regression":
            targets = [_finite(value) for value in targets]
        elif scorer == "composition":
            try:
                targets = [[_finite(value) for value in row] for row in targets]
            except TypeError as exc:
                raise ValueError("composition references must be component vectors") from exc
            if not targets[0] or any(len(row) != len(targets[0]) or min(row) < 0 for row in targets):
                raise ValueError("composition references require equal nonempty nonnegative component vectors")
        elif scorer in {"binary", "residue_probabilities"}:
            if scorer == "residue_probabilities" and any(not row for row in targets):
                raise ValueError("each training protein must have eligible residue labels")
            flat = targets if scorer == "binary" else [value for row in targets for value in row]
            if not flat or any(_finite(value) not in (0, 1) for value in flat):
                raise ValueError("training binary references must be finite 0/1 labels")
            targets = [float(value) for value in flat]
        elif scorer == "classification":
            targets = [_label(value) for value in targets]
            if isinstance(label_universe, (str, bytes, Mapping)):
                raise ValueError("label_universe must be a sequence of labels")
            labels = list(dict.fromkeys(targets)) if label_universe is None else list(label_universe)
            if not labels or len({_label(value) for value in labels}) != len(labels):
                raise ValueError("label_universe must be nonempty and unique")
            if any(value not in labels for value in targets):
                raise ValueError("training label absent from declared label_universe")
            counts = Counter(targets)
            self._labels, self._counts = labels, [counts[value] for value in labels]
        elif scorer in {"set_recovery", "sequence_recovery"}:
            if any(isinstance(row, (str, bytes, Mapping)) for row in targets):
                raise ValueError("training annotations must be normalized sets or token sequences")
            if scorer == "sequence_recovery" and any(not isinstance(row, Sequence) for row in targets):
                raise ValueError("training action sequences require a declared token order")
            targets = [[_label(value) for value in row] for row in targets]
            if scorer == "set_recovery":
                targets = [set(row) for row in targets]
        else:
            raise ValueError("this strategy uses permitted query inputs rather than fit targets")
        if label_universe is not None and scorer != "classification":
            raise ValueError("label_universe applies only to classification controls")
        self._targets, self._training_ids = targets, frozenset(ids)
        if self.strategy in {"train_mean", "train_median"}:
            statistic = mean if self.strategy == "train_mean" else median
            self._constant = ([statistic(component) for component in zip(*targets)]
                              if scorer == "composition" else statistic(targets))
        elif self.strategy == "train_prior" and scorer != "classification":
            self._constant = mean(targets)
        self._fitted = True
        return self

    def predict(self, *, instance_ids: Sequence[Any],
                candidate_ids: Sequence[Sequence[Any]] | None = None,
                initial_values: Sequence[float] | None = None,
                lengths: Sequence[int] | None = None) -> list[Any]:
        """Predict without receiving test labels, responses or references.

        :param instance_ids: Unique query IDs, disjoint from training IDs.
        :param candidate_ids: Available candidate IDs for random ordering only.
        :param initial_values: Last permitted observed scalar for persistence.
        :param lengths: Forecast horizon or eligible residue count per query.
        :return: Predictions aligned to ``instance_ids``.
        :raises ValueError: For overlap, missing fit state or unsuitable inputs.
        """
        ids = _identifiers(instance_ids, "instance_ids")
        if self._training_ids.intersection(ids):
            raise ValueError("prediction instance also appears in the training partition")
        if self.strategy.startswith("train_") or self.strategy == "uniform_labels":
            if not self._fitted:
                raise ValueError("fit a nonempty training partition before prediction")
        expected = set()
        if self.strategy == "random_order":
            expected.add("candidate_ids")
        if self.strategy == "causal_persistence":
            expected.add("initial_values")
            if self.protocol.scorer == "trajectory":
                expected.add("lengths")
        if self.protocol.scorer == "residue_probabilities":
            expected.add("lengths")
        supplied = {name: value for name, value in {
            "candidate_ids": candidate_ids, "initial_values": initial_values,
            "lengths": lengths}.items() if value is not None}
        if set(supplied) != expected:
            raise ValueError(f"this strategy requires only these query inputs: {sorted(expected)}")
        if any(isinstance(value, (str, bytes, Mapping)) for value in supplied.values()):
            raise ValueError("permitted query inputs must be aligned sequences, not mappings or strings")
        if any(len(value) != len(ids) for value in supplied.values()):
            raise ValueError("permitted query inputs must align to instance_ids")
        if lengths is not None and any(type(length) is not int or length < 1 for length in lengths):
            raise ValueError("lengths must be positive integers")
        result = []
        for index, instance_id in enumerate(ids):
            key = json.dumps([self.seed, self.protocol.task_id, self.protocol.variant,
                              instance_id], ensure_ascii=False).encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
            if self.strategy == "random_order":
                value = sorted(_identifiers(candidate_ids[index], "candidate_ids"),
                               key=lambda item: json.dumps([type(item).__name__, item]))
                rng.shuffle(value)
            elif self.strategy == "causal_persistence":
                value = _finite(initial_values[index])
                if lengths is not None:
                    value = [value] * lengths[index]
            elif self.protocol.scorer == "residue_probabilities":
                value = [self._single(rng) for _ in range(lengths[index])]
            else:
                value = self._single(rng)
            result.append(value)
        return result

    def _single(self, rng: random.Random) -> Any:
        if self.strategy == "zero_change":
            return 0.0
        if self.strategy in {"train_mean", "train_median"}:
            return deepcopy(self._constant)
        if self.strategy == "train_sample":
            return deepcopy(rng.choice(self._targets))
        if self.strategy == "train_majority":
            return self._labels[self._counts.index(max(self._counts))]
        if self.strategy == "train_prior":
            if self.protocol.scorer == "classification":
                return rng.choices(self._labels, weights=self._counts, k=1)[0]
            return self._constant
        if self.strategy == "uniform_labels":
            return rng.choice(self._labels)
        if self.strategy == "uniform_probability":
            return .5
        if self.strategy == "random_probability":
            return rng.random()
        if self.strategy == "empty":
            return set() if self.protocol.scorer == "set_recovery" else []
        if self.strategy == "one_cluster":
            return 0
        if self.strategy == "random_clusters":
            return rng.randrange(self.n_clusters)
        raise ValueError("strategy requires explicit permitted query inputs")
