"""Keep manuscript quantities tied to the scientific inventory."""
import copy
import csv
import json

import pytest

from synrxn.reconstruction.presentation.manuscript import (
    build_facts, build_license_ledger, generate, render_license_key, render_tasks,
)

from synrxn.reconstruction.paths import REPO, physical_path


def inputs():
    return [json.loads(physical_path(path, repo=REPO).read_text()) for path in (
        "kdd/task_registry.json", "kdd/hierarchy/hierarchy.json",
        "kdd/coverage/release_manifest.json", "kdd/case_studies/ncs_yield/report.json")]


def test_manuscript_counts_sources_once_and_retains_exclusion_denominators():
    facts = build_facts(*inputs())
    assert (facts["families"], facts["tasks"], facts["catalogue_table_views"]) == (12, 41, 105)
    assert facts["task_table_bindings"] == 146
    assert facts["stored_rows"] == 19611896
    assert facts["primary_stored_rows_without_whole_frea_corpus"] == 5486114
    assert facts["release_admitted_catalogue_views"] == 104
    assert facts["ncs_positive"] + facts["ncs_negative"] == facts["ncs_canonical_rows"] == 27438
    assert "Pending" in facts["predictive_results"]


def test_conflicting_reference_identity_cannot_enter_generated_paper():
    values = copy.deepcopy(inputs())
    values[1]["families"][0]["tasks"][0]["tables"][0]["key"] = "invented/unknown"
    with pytest.raises(ValueError, match="task bindings"):
        build_facts(*values)


def test_generated_quantities_and_task_table_are_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    facts = generate(REPO, first)
    generate(REPO, second)
    assert len(facts["input_sha256"]) == 4
    for name in ("facts.json", "stats.tex", "tasks.tex", "licenses.tex",
                 "source_licenses.json", "source_licenses.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    registry, _, release, _ = inputs()
    table = render_tasks(registry, build_license_ledger(registry, release))
    assert sum(line.endswith(r" \\") and not line.startswith("\\")
               for line in table.splitlines()) == len(registry["tasks"])


def test_source_license_ledger_preserves_joint_terms_and_catalogue_holds():
    registry, _, release, _ = inputs()
    ledger = build_license_ledger(registry, release)
    sources = {(t["namespace"], t["key"]): t for t in ledger["tables"]}
    assert len(sources) == 105
    assert len(ledger["license_key"]) == 18
    assert sum(t["release_admitted"] for t in sources.values()) == 104
    assert sum(len(t["task_ids"]) for t in sources.values()) == 146
    assert sum(len(t["primary_task_ids"]) for t in sources.values()) == 137
    # The effective NCS data grant takes precedence over the old code-license field.
    ncs = sources[("canonical", "yield/buchwald_hartwig_ood")]
    assert ncs["license_expression"] == "CC-BY-4.0"
    golden = sources[("canonical", "aam/golden")]
    assert golden["license_expression"] == "GPL-3.0-only AND LGPL-2.1-only AND MIT"
    held = [t for t in sources.values() if not t["release_admitted"]]
    assert [(t["namespace"], t["key"], t["license_expression"], t["license_code"])
            for t in held] == [("canonical", "synthesis/da", None, "H")]
    assert held[0]["primary_task_ids"] == ["forward_prediction", "retrosynthesis"]
    # Four quarantined enzyme views in the release audit are outside this catalogue.
    assert not any(t["namespace"] == "coverage_quarantine" for t in sources.values())
    assert "AND" in render_license_key(ledger)
    task_rows = render_tasks(registry, ledger).splitlines()
    assert sum(" & H" in row or ", H" in row for row in task_rows) == 2


def test_license_generation_rejects_disagreement_instead_of_inventing_grant():
    registry, _, release, _ = inputs()
    registry["tasks"][0]["bindings"][0]["release_license_expression"] = "MIT"
    with pytest.raises(ValueError, match="source-rights manifest disagree"):
        build_license_ledger(registry, release)
    registry, _, release, _ = inputs()
    release["release_tables"][0]["license_expression"] = None
    with pytest.raises(ValueError, match="no license expression"):
        build_license_ledger(registry, release)


def test_exported_csv_retains_verbatim_expression_and_source_sha(tmp_path):
    generate(REPO, tmp_path)
    ledger = json.loads((tmp_path / "source_licenses.json").read_text())
    with (tmp_path / "source_licenses.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(ledger["tables"]) == 105
    json_by_key = {(row["namespace"], row["key"]): row for row in ledger["tables"]}
    for row in rows:
        source = json_by_key[(row["namespace"], row["key"])]
        assert row["license_expression"] == (source["license_expression"] or "")
        assert row["sha256"] == source["sha256"]
        assert len(row["sha256"]) == 64
        assert json.loads(row["task_ids"]) == source["task_ids"]
