"""Analytical evaluator fixtures, distinct from empirical baseline results.

Tests intentionally include oracle copies and malformed predictions only to
check evaluator semantics. Those outputs are never deployable dummy controls.
"""

import math

import pytest

from synrxn.baseline.dummy import (
    DummyPredictor, get_dummy_protocol, list_dummy_protocols,
)
from synrxn.metrics import evaluate_task, get_task_metrics, list_task_metrics


def test_discovery_matches_every_task_variant_and_explains_unsupported_types():
    protocols = list_dummy_protocols()
    assert set(protocols) == set(list_task_metrics())
    for task_id, variants in protocols.items():
        task = get_task_metrics(task_id)
        assert set(variants) == set(task.variants)
        for variant, protocol in variants.items():
            assert protocol.variant == variant
            assert protocol.family_id == task.family_id
            assert protocol.requirements and protocol.qualification
            if protocol.status == "specialized_adapter_required":
                assert not protocol.strategies
                with pytest.raises(ValueError, match="unsupported"):
                    DummyPredictor(task_id, "random", variant=variant)
    for task_id in ("atom_mapping", "forward_prediction", "mechanistic_prediction",
                    "ts_geometry", "process_control", "reactive_potential"):
        assert get_dummy_protocol(task_id).status == "specialized_adapter_required"
    with pytest.raises(KeyError):
        get_dummy_protocol("yield_conversion", "made_up")


def _default_fixture(task_id):
    """Construct independent synthetic train and held-out input schemas."""
    kind = get_dummy_protocol(task_id).scorer
    config = {
        "regression": ("train_mean", [1., 3.], [1., 3.], {"unit": "native"}),
        "classification": ("train_majority", ["a", "b"], ["a", "b"], {}),
        "binary": ("train_prior", [0, 1], [0, 1], {}),
        "set_recovery": ("empty", None, [{"a"}, {"b"}], {}),
        "sequence_recovery": ("empty", None, [["add"], ["heat"]], {}),
        "composition": ("train_mean", [[.7, .1], [.5, .3]], [[.6, .2], [.6, .2]], {"unit": "fraction"}),
        "ranked_retrieval": ("random_order", None, [{"a": 1, "b": 0}] * 2, {}),
        "observed_optimization": ("random_order", None, [{"a": 1, "b": 3}] * 2, {"unit": "native", "budgets": (1, 2)}),
        "clustering": ("one_cluster", None, [0, 1], {}),
        "trajectory": ("causal_persistence", None, [[1, 2, 3], [2, 3, 4]], {"unit": "native", "times": [[0, 1, 2]] * 2, "time_unit": "s"}),
        "residue_probabilities": ("train_prior", [[0, 1], [0, 1]], [[0, 1], [0, 1]], {}),
    }
    strategy, training, reference, context = config[kind]
    model = DummyPredictor(task_id, strategy, seed=7)
    if training is not None:
        model.fit(training, training_ids=["train-0", "train-1"])
    inputs = {"instance_ids": ["held-out-0", "held-out-1"]}
    if kind in {"ranked_retrieval", "observed_optimization"}:
        inputs["candidate_ids"] = [["a", "b"], ["a", "b"]]
    if kind == "trajectory":
        inputs.update(initial_values=[1, 2], lengths=[3, 3])
    if kind == "residue_probabilities":
        inputs["lengths"] = [2, 2]
    return model, reference, inputs, context


@pytest.mark.parametrize("task_id", [
    task_id for task_id in list_task_metrics()
    if get_dummy_protocol(task_id).status == "generic_control"
])
def test_each_supported_default_dispatches_from_dummy_to_task_evaluator(task_id):
    model, reference, inputs, context = _default_fixture(task_id)
    prediction = model.predict(**inputs)
    result = evaluate_task(task_id, reference, prediction, **context)
    assert result["task_id"] == task_id
    assert result["metrics"]["coverage"] == 1
    assert len(prediction) == len(reference)


def test_mean_and_median_use_only_training_targets_with_analytic_errors():
    mean_model = DummyPredictor("reaction_rates", "train_mean").fit(
        [1, 3], training_ids=["train-1", "train-2"])
    prediction = mean_model.predict(instance_ids=["test-1", "test-2"])
    result = evaluate_task("reaction_rates", [1, 3], prediction, unit="log10 native rate")
    assert prediction == [2, 2]
    assert result["metrics"]["mae"] == result["metrics"]["rmse"] == 1
    median_model = DummyPredictor("reaction_rates", "train_median").fit(
        [0, 1, 100], training_ids=["train-1", "train-2", "train-3"])
    assert median_model.predict(instance_ids=["test-1"]) == [1]
    with pytest.raises(TypeError):
        mean_model.predict(instance_ids=["test-1"], reference=[999])
    with pytest.raises(ValueError, match="training partition"):
        mean_model.predict(instance_ids=["train-1"])


def test_mutation_zero_change_requires_no_fit_and_has_native_delta_errors():
    assert "zero_change" in get_dummy_protocol("mutation_effects").strategies
    assert "zero_change" not in get_dummy_protocol("reaction_rates").strategies
    model = DummyPredictor("mutation_effects", "zero_change")
    prediction = model.predict(instance_ids=["mutant-1", "mutant-2", "mutant-3"])
    assert prediction == [0., 0., 0.]
    result = evaluate_task("mutation_effects", [-2., 3., 0.], prediction,
                           unit="native activity delta")["metrics"]
    assert result["coverage"] == 1
    assert result["mae"] == pytest.approx(5 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(13 / 3))
    assert result["direction_accuracy"] == 0 and result["direction_denominator"] == 2


def test_component_controls_preserve_native_totals_and_do_not_alias_training():
    training = [[.7, .1], [.5, .3]]
    model = DummyPredictor("product_distribution", "train_mean").fit(
        training, training_ids=["train-1", "train-2"])
    training[0][0] = 999
    prediction = model.predict(instance_ids=["test-1", "test-2"])
    assert prediction[0] == pytest.approx([.6, .2])
    assert sum(prediction[0]) == pytest.approx(.8)
    prediction[0][0] = 999
    assert model.predict(instance_ids=["test-1"])[0] == pytest.approx([.6, .2])
    model = DummyPredictor("product_distribution", "train_median").fit(
        [[1, 2], [3, 6]], training_ids=["train-1", "train-2"])
    assert model.predict(instance_ids=["test-1"]) == [[2, 4]]


def test_probability_prior_has_known_discrimination_and_calibration_scores():
    model = DummyPredictor("reaction_success", "train_prior").fit(
        [0, 1], training_ids=["train-1", "train-2"])
    prediction = model.predict(instance_ids=["test-1", "test-2"])
    result = evaluate_task("reaction_success", [0, 1], prediction)["metrics"]
    assert prediction == [.5, .5]
    assert result["accuracy"] == result["auroc"] == result["average_precision"] == .5
    assert result["brier_score"] == .25
    assert result["log_loss"] == pytest.approx(math.log(2))
    assert result["expected_calibration_error"] == 0


def test_classification_uses_declared_or_training_labels_and_stable_majority_ties():
    model = DummyPredictor("reaction_classification", "train_majority").fit(
        ["a", "b"], training_ids=["train-1", "train-2"])
    assert model.predict(instance_ids=["test-1"]) == ["a"]
    model = DummyPredictor("reaction_classification", "uniform_labels").fit(
        ["a", "a"], training_ids=["train-1", "train-2"], label_universe=["a", "b"])
    assert set(model.predict(instance_ids=[f"test-{i}" for i in range(20)])) <= {"a", "b"}
    model = DummyPredictor("reaction_classification", "train_prior").fit(
        ["a", "a"], training_ids=["train-1", "train-2"], label_universe=["a", "b"])
    assert model.predict(instance_ids=["test-1"]) == ["a"]
    with pytest.raises(ValueError, match="sequence of labels"):
        model.fit(["a", "b"], training_ids=["train-1", "train-2"], label_universe="ab")


def test_random_predictions_are_reproducible_by_id_across_batches_and_order():
    model = DummyPredictor("reaction_success", "random_probability", seed=19)
    ids = [f"test-{i}" for i in range(12)]
    complete = dict(zip(ids, model.predict(instance_ids=ids)))
    reverse = list(reversed(ids))
    assert dict(zip(reverse, model.predict(instance_ids=reverse))) == complete
    assert model.predict(instance_ids=ids[:3]) == [complete[item] for item in ids[:3]]
    assert all(0 <= value <= 1 for value in complete.values())
    assert complete != dict(zip(ids, DummyPredictor(
        "reaction_success", "random_probability", seed=20).predict(instance_ids=ids)))


def test_random_candidate_order_uses_only_available_ids_and_not_source_order():
    model = DummyPredictor("condition_ranking", "random_order", seed=3)
    first = model.predict(instance_ids=["panel-1", "panel-2"],
                          candidate_ids=[["a", "b", "c"], ["a", "b", "c"]])
    single = model.predict(instance_ids=["panel-1"], candidate_ids=[["c", "a", "b"]])
    assert single[0] == first[0]
    assert set(first[0]) == {"a", "b", "c"} and len(first[0]) == 3
    result = evaluate_task("condition_ranking", [{"a": 1, "b": 2, "c": 3}],
                           single, budgets=(3,), unit="native response")["metrics"]
    assert result["budgets"][3]["mean_observed_regret"] == 0
    with pytest.raises(ValueError, match="unique"):
        model.predict(instance_ids=["panel-1"], candidate_ids=[["a", "a"]])
    with pytest.raises(ValueError, match="sequence of identifiers"):
        model.predict(instance_ids=["panel-1"], candidate_ids=[{"a": 3, "b": 1}])
    with pytest.raises(ValueError, match="aligned sequences"):
        model.predict(instance_ids=["panel-1"], candidate_ids={"panel-1": ["a", "b"]})
    with pytest.raises(TypeError):
        model.predict(instance_ids=["panel-1"], candidate_ids=[["a"]], oracle={"a": 3})


def test_causal_persistence_uses_only_supplied_observed_state():
    model = DummyPredictor("reaction_progress_monitoring", "causal_persistence")
    assert model.predict(instance_ids=["trajectory-1"], initial_values=[.2], lengths=[3]) == [[.2] * 3]
    scalar = DummyPredictor("reaction_progress_monitoring", "causal_persistence", variant="next_step")
    assert scalar.predict(instance_ids=["trajectory-1"], initial_values=[.2]) == [.2]
    with pytest.raises(ValueError, match="requires only"):
        model.predict(instance_ids=["trajectory-1"], initial_values=[.2])


@pytest.mark.parametrize("task_id,training", [
    ("reaction_rates", [float("nan")]),
    ("reaction_rates", [float("inf")]),
    ("reaction_rates", [None]),
    ("product_distribution", [[1, -.1]]),
    ("product_distribution", [[float("nan"), 0]]),
])
def test_invalid_training_references_rejected(task_id, training):
    with pytest.raises(ValueError):
        DummyPredictor(task_id, "train_mean").fit(training, training_ids=["train-1"])


def test_empty_fit_missing_fit_and_invalid_inputs_are_rejected():
    model = DummyPredictor("reaction_rates", "train_mean")
    with pytest.raises(ValueError):
        model.fit([], training_ids=[])
    with pytest.raises(ValueError, match="fit"):
        model.predict(instance_ids=["test-1"])
    with pytest.raises(ValueError, match="equal nonzero length"):
        model.fit([1], training_ids=["train-1", "train-2"])
    with pytest.raises(ValueError, match="unique"):
        model.fit([1, 2], training_ids=["train-1", "train-1"])
    with pytest.raises(ValueError):
        DummyPredictor("reaction_success", "train_prior").fit([2], training_ids=["train-1"])


@pytest.mark.parametrize("task_id,training", [
    ("reaction_rates", [1, 4, 9]),
    ("product_distribution", [[.7, .1], [.5, .3], [.2, .4]]),
    ("reaction_center", [{"edit-a"}, set(), {"edit-b"}]),
    ("procedure_generation", [["add"], ["heat"], ["wash"]]),
])
def test_empirical_samples_are_drawn_only_from_training_annotations(task_id, training):
    model = DummyPredictor(task_id, "train_sample", seed=3).fit(
        training, training_ids=["train-1", "train-2", "train-3"])
    predictions = model.predict(instance_ids=[f"test-{i}" for i in range(10)])
    assert all(value in training for value in predictions)


def test_identity_oracle_and_corruption_are_evaluator_fixtures_only():
    reference = [1., 3.]
    identity = evaluate_task("reaction_rates", reference, reference, unit="native")["metrics"]
    worse = evaluate_task("reaction_rates", reference, [4., 6.], unit="native")["metrics"]
    missing = evaluate_task("reaction_rates", reference, [None, None], unit="native")["metrics"]
    assert identity["mae"] == 0 and worse["mae"] == 3
    assert missing["mae"] is None and missing["coverage"] == 0
    invalid = evaluate_task("reaction_success", [0, 1], [-1, 2])["metrics"]
    assert invalid["coverage"] == 0 and invalid["accuracy"] == 0
    with pytest.raises(ValueError, match="unsupported"):
        DummyPredictor("reaction_rates", "oracle")
