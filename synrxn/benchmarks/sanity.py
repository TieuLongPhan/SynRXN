"""Synthetic metric checks, independent of datasets, fingerprints and training.

Reference copies are evaluator fixtures, never empirical model predictions.
The control fixture supplies traces to a scorer; it does not run a controller.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

from synrxn.metrics import evaluate_task, get_task_metrics, list_task_metrics


class _BackendUnavailable(RuntimeError):
    """An explicitly identified optional chemistry capability is unavailable."""


def _check_backend(scorer):
    if scorer == "reference_contract":
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise _BackendUnavailable("Chemical reference fixtures require RDKit") from exc
        if not hasattr(Chem, "MolFromSmiles"):
            raise _BackendUnavailable("RDKit molecule parsing is unavailable")
    if scorer == "mechanism":
        from synrxn.metrics.mechanism import _backend

        try:
            _backend()
        except RuntimeError as exc:
            if str(exc).startswith(("Mechanism metrics require", "SynKit comparator drops")):
                raise _BackendUnavailable(str(exc)) from exc
            raise


@dataclass(frozen=True)
class _Case:
    reference: object
    perfect: object
    wrong: object
    missing: object
    context: dict
    metric: str
    ideal: float
    coverage: str = "coverage"


def _chemical_case(task_id):
    from .contracts import MAPPING_POLICY, OUTCOME_POLICY, ReferenceRecord, TaskContract

    mapping = task_id == "atom_mapping"
    kind = "mapping" if mapping else "forward"
    contract = TaskContract(
        "synthetic." + kind, "1", kind,
        ("reactants", "products") if mapping else ("reactants",),
        "complete_correspondence" if mapping else "reported_product_multiset",
        MAPPING_POLICY if mapping else OUTCOME_POLICY,
        ("test_review",), status="runnable",
    )
    reference = ReferenceRecord(
        reference_id="fixture.reference", observation_id="fixture.observation",
        revision=1, source_record_id="fixture.source", evidence_tier="test_review",
        resolution="accepted", evidence_locators=("synthetic evaluator fixture",),
        task_eligibility={kind: True}, review_events=({"reviewer": "fixture"},),
        **({"accepted_correspondences": ((0, 1),)} if mapping
           else {"accepted_outcomes": (("CC=O",),)}),
    )
    inputs = {"reactants": "CO", "products": "CO"} if mapping else {"reactants": "CCO"}
    instances = [{"instance_id": "fixture.instance", "reference_id": reference.reference_id,
                  "reference_revision": 1, "observation_id": reference.observation_id,
                  "inputs": inputs}]

    def prediction(value):
        return [{"instance_id": "fixture.instance", "status": "ok", "candidates": [value]}]

    return _Case(
        [reference], prediction([0, 1] if mapping else "CC=O"),
        prediction([1, 0] if mapping else "C"), [],
        {"instances": instances, "contract": contract}, "top1_accuracy", 1.,
        "valid_output_coverage",
    )


def _mechanism_case():
    from synkit.Mechanism import mechanism_from_legacy_epd

    record = mechanism_from_legacy_epd(
        "[NH3+:2][H:4].[O-:1][CH3:3]>>[NH3:2].[O:1]([CH3:3])[H:4]",
        [["LP-/Sigma+", [1], [1, 4]], ["Sigma-/LP+", [2, 4], [2]]],
    )
    return _Case(
        {"fixture": [record]}, {"fixture": [record]},
        {"fixture": [replace(record, steps=())]}, {}, {"top_k": (1,)},
        "top_k.1.event_recovery", 1., "prediction_coverage",
    )


def _fixture(task_id, scorer):
    scalar = {"unit": "synthetic-native-unit"}
    if scorer == "reference_contract":
        return _chemical_case(task_id)
    if scorer == "mechanism":
        return _mechanism_case()
    if scorer == "regression":
        return _Case([2., 4.], [2., 4.], [0., 0.], [None, None], scalar, "mae", 0.)
    if scorer == "classification":
        return _Case(["a", "b"], ["a", "b"], ["b", "a"], [None, None], {}, "accuracy", 1.)
    if scorer == "binary":
        return _Case([0, 1], [0., 1.], [1., 0.], [None, None], {}, "brier_score", 0.)
    if scorer == "set_recovery":
        return _Case([{"a"}, {"b"}], [{"a"}, {"b"}], [set(), set()], [None, None], {}, "observed_recall", 1.)
    if scorer == "ranked_retrieval":
        return _Case([{"a": 1., "b": 0., "c": 0.}], [["a", "b", "c"]],
                     [["b", "c", "a"]], [[]], {"top_k": (1,)}, "top_k.1.recall", 1.)
    if scorer == "observed_optimization":
        return _Case([{"a": 1., "b": 3., "c": 2.}], [["b", "c", "a"]],
                     [["a", "c", "b"]], [[]], {**scalar, "budgets": (1,)},
                     "budgets.1.mean_observed_regret", 0.)
    if scorer == "sequence_recovery":
        return _Case([["add", "heat"]], [["add", "heat"]], [["cool"]], [None], {}, "exact_match", 1.)
    if scorer == "composition":
        return _Case([[.2, .6, .1]], [[.2, .6, .1]], [[0., 0., 0.]], [None],
                     {"unit": "fraction"}, "component_mae", 0.)
    if scorer == "clustering":
        return _Case([0, 0, 1, 1], [9, 9, 8, 8], [0, 1, 0, 1], [None] * 4, {}, "adjusted_rand_index", 1.)
    if scorer == "geometry":
        positions = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]
        wrong = deepcopy(positions)
        wrong[-1][-1] = 6.
        return _Case([positions], [positions], [wrong], [None], {"unit": "angstrom"}, "mean_aligned_rmsd", 0.)
    if scorer in {"trajectory", "control"}:
        context = {"times": [[0., 1., 2.]], "unit": "mol/L", "time_unit": "s"}
        if scorer == "control":
            context.update(actions=[[0., 0., 0.]], action_unit="native-action", action_bounds=(0., 1.))
        return _Case([[0., 1., 2.]], [[0., 1., 2.]], [[2., 2., 2.]], [None], context, "macro_mae", 0.)
    if scorer == "residue_probabilities":
        return _Case([[0, 1]], [[0., 1.]], [[1., 0.]], [None], {}, "protein_macro_average_precision", 1.)
    if scorer == "energy_forces":
        truth = {"energy": [1., 2.], "forces": [[[1., 0., 0.]], [[0., 1., 0.]]]}
        wrong = {"energy": [0., 0.], "forces": [[[0., 0., 0.]], [[0., 0., 0.]]]}
        return _Case(truth, truth, wrong, {"energy": [None, None], "forces": None},
                     {"energy_unit": "eV", "force_unit": "eV/angstrom"}, "energy.mae", 0.)
    raise ValueError("No synthetic fixture for scorer: " + scorer)


def _value(record, path):
    for key in path.split("."):
        record = record[key] if key in record else record[int(key)]
    return record


def run_metric_sanity(task_ids=None):
    """Check perfect, wrong and missing outputs on synthetic task fixtures.

    :param task_ids: Optional scientific task IDs; defaults to all registered tasks.
    :return: Per-task checks, explicit backend blocks and an overall status.

    Only default variants are covered by this command. Source eligibility,
    empirical null scores and nondefault variants require their own checks.
    """
    selected = list(list_task_metrics() if task_ids is None else task_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one task, without duplicates")
    rows = []
    for task_id in selected:
        profile = get_task_metrics(task_id)
        scorer = profile.variants[profile.default_variant].scorer
        row = {"task_id": task_id, "family_id": profile.family_id,
               "variant": profile.default_variant, "scorer": scorer}
        try:
            _check_backend(scorer)
            case = _fixture(task_id, scorer)
            outputs = [evaluate_task(task_id, case.reference, prediction, **case.context)["metrics"]
                       for prediction in (case.perfect, case.wrong, case.missing)]
            perfect, wrong = [_value(score, case.metric) for score in outputs[:2]]
            coverage = _value(outputs[2], case.coverage)
            missing = _value(outputs[2], case.metric)
            zero_on_missing = scorer in {
                "classification", "set_recovery", "ranked_retrieval",
                "sequence_recovery", "reference_contract", "mechanism",
            }
            checks = {
                "perfect_score": perfect is not None and math.isclose(perfect, case.ideal, abs_tol=1e-10),
                "wrong_is_worse": wrong is not None and (
                    wrong > perfect + 1e-10 if case.ideal == 0 else wrong < perfect - 1e-10),
                "missing_coverage_zero": coverage == 0,
                "missing_score": missing == 0 if zero_on_missing else missing is None,
            }
            if scorer == "energy_forces":
                checks["missing_force_frame_coverage_zero"] = outputs[2]["force_configuration_coverage"]["coverage"] == 0
            row.update(status="passed" if all(checks.values()) else "failed",
                       metric=case.metric, perfect=perfect, wrong=wrong,
                       missing_score=missing, missing_coverage=coverage, checks=checks)
        except _BackendUnavailable as exc:
            row.update(status="blocked", reason=str(exc))
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("passed", "failed", "blocked")}
    return {
        "schema": "synrxn.metric-sanity.v1", "scope": "synthetic-default-variant-fixtures",
        "status": "failed" if counts["failed"] else "blocked" if counts["blocked"] else "passed",
        "summary": {"tasks": len(rows), **counts}, "tasks": rows,
        "qualification": "No datasets, encodings, model training or empirical performance. Reference-copy/oracle outputs and supplied control traces are evaluator fixtures only. Source-specific validity and split qualification remain separate.",
    }


def main(argv=None):
    """Write a synthetic-only metric report; blocked checks return nonzero.

    :param argv: Optional command-line arguments.
    :return: Zero only if every selected task check passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", dest="tasks", help="Scientific task ID; repeat to select several")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_metric_sanity(args.tasks)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], **report["summary"],
                      "output": str(args.output) if args.output else None}))
    for row in report["tasks"]:
        if row["status"] != "passed":
            print(json.dumps(row, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
