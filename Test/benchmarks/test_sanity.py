"""Synthetic evaluator checks remain separate from empirical baselines."""

import json

import pytest

from synrxn.benchmarks.sanity import main, run_metric_sanity


def test_known_regression_ranking_and_composition_scores():
    result = run_metric_sanity(["reaction_rates", "condition_ranking", "product_distribution"])
    assert result["status"] == "passed", result
    rows = {row["task_id"]: row for row in result["tasks"]}
    assert rows["reaction_rates"]["perfect"] == 0
    assert rows["reaction_rates"]["wrong"] == 3
    assert rows["condition_ranking"]["wrong"] == 2
    assert rows["product_distribution"]["wrong"] == pytest.approx(.3)
    assert all(row["missing_coverage"] == 0 for row in rows.values())
    json.dumps(result, allow_nan=False)


def test_unknown_or_duplicate_task_selections_are_not_silent():
    with pytest.raises(KeyError):
        run_metric_sanity(["not_a_task"])
    with pytest.raises(ValueError, match="duplicates"):
        run_metric_sanity(["reaction_rates", "reaction_rates"])
    with pytest.raises(ValueError, match="at least one"):
        run_metric_sanity([])


def test_missing_chemical_backend_is_reported_as_blocked(monkeypatch):
    from synrxn.benchmarks import sanity

    original = sanity._check_backend

    def check_backend(scorer):
        if scorer == "mechanism":
            raise sanity._BackendUnavailable("backend unavailable")
        return original(scorer)

    monkeypatch.setattr(sanity, "_check_backend", check_backend)
    result = sanity.run_metric_sanity(["mechanistic_prediction", "reaction_rates"])
    assert result["status"] == "blocked"
    assert result["summary"] == {"tasks": 2, "passed": 1, "failed": 0, "blocked": 1}


def test_scorer_bugs_are_failures_instead_of_backend_blocks(monkeypatch):
    from synrxn.benchmarks import sanity

    def broken(*args, **kwargs):
        raise RuntimeError("implementation failure")

    monkeypatch.setattr(sanity, "_check_backend", lambda scorer: None)
    monkeypatch.setattr(sanity, "evaluate_task", broken)
    result = sanity.run_metric_sanity(["forward_prediction"])
    assert result["status"] == "failed"
    assert "implementation failure" in result["tasks"][0]["reason"]


def test_zero_coverage_cannot_hide_perfect_scores_for_missing_outputs(monkeypatch):
    from synrxn.benchmarks import sanity

    evaluate = sanity.evaluate_task

    def wrong_missing_score(task, reference, prediction, **context):
        result = evaluate(task, reference, prediction, **context)
        if result["metrics"]["coverage"] == 0:
            result["metrics"]["accuracy"] = 1
        return result

    monkeypatch.setattr(sanity, "evaluate_task", wrong_missing_score)
    result = sanity.run_metric_sanity(["reaction_classification"])
    assert result["status"] == "failed"
    assert not result["tasks"][0]["checks"]["missing_score"]


def test_cli_does_not_hide_incomplete_checks(tmp_path, capsys):
    path = tmp_path / "sanity.json"
    assert main(["--task", "reaction_rates", "--output", str(path)]) == 0
    assert json.loads(path.read_text())["scope"] == "synthetic-default-variant-fixtures"
    assert json.loads(capsys.readouterr().out)["passed"] == 1
