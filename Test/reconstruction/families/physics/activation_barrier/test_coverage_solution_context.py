"""Guard solution-condition retention, target isolation and parent overlap."""
import csv
import copy
import gzip
import hashlib
import io
import json

from synrxn.reconstruction.families.physics.activation_barrier import _solution_context as adapter
from synrxn.reconstruction.coverage.rights import source_has_evidence, table_has_evidence

from synrxn.reconstruction.paths import REPO as ROOT, source_path
BASE = source_path(ROOT / "Data/coverage_expansion/solution_context")


def records(identifier):
    path = source_path(f"Data/coverage_expansion/solution_context/{identifier}.csv")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_solution_context_retains_native_quantities_and_conditions():
    snar = records("snar_solution_context")
    cyclo = records("cycloadd_solution_context")
    assert len(snar) == 503 and len(cyclo) == 5269
    assert len({r["solvent"] for r in snar}) == 41
    assert len({r["temperature_K"] for r in snar}) == 13
    assert min(float(r["temperature_K"]) for r in snar) == 273
    assert max(float(r["temperature_K"]) for r in snar) == 468.4
    assert {(r["solvent"], r["temperature_K"]) for r in cyclo} == {("water", "298.15")}
    for key, rows, targets in [("snar", snar, {"ea": "activation_free_energy_kcal_mol"}),
                               ("cycloadd", cyclo, {"G_act": "activation_free_energy_kcal_mol", "G_r": "reaction_free_energy_kcal_mol"})]:
        with gzip.open(source_path(ROOT / "Data/property" / (key + ".csv.gz")), "rt") as handle:
            parent = {r["r_id"]: r for r in csv.DictReader(handle)}
        assert {r["parent_record_id"].split(":", 1)[1] for r in rows} == set(parent)
        for row in rows:
            original = parent[row["parent_record_id"].split(":", 1)[1]]
            for old, new in targets.items():
                assert row[new] == original[old]


def test_condition_variants_share_groups_and_no_targets_enter_inputs():
    catalog = json.loads((BASE / "catalog.json").read_text())
    for table in catalog["tables"]:
        rows = records(table["id"])
        groups = {}
        for row in rows:
            groups.setdefault(row["reaction_group"], set()).add(row["proposed_split"])
        assert all(len(splits) == 1 for splits in groups.values())
        assert table["overlapping_view"] is True
        assert table["role"] == "overlapping_view"
        assert table["derived_view_of"] == table["overlap_parent"]
        assert table["independent_observations_added"] == 0
        for task in table["task_fields"].values():
            assert {"solvent", "temperature_K"} <= set(task["inputs"])
            assert not set(task["inputs"]) & set(task["targets"])
            assert not any("energy" in key for key in task["inputs"])
            assert {"reaction_group", "parent_dataset", "parent_record_id"} <= set(task["group_fields"])
        if table["id"] == "snar_solution_context":
            assert len(groups) == 346 < len(rows)
    # Chemical atom order, component order and map labels cannot split a parent.
    assert adapter.reaction_group("[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]") == adapter.reaction_group("[OH-:7].BrC>>[Br-].OC")


def test_native_context_reconstruction_and_source_specific_rights():
    adapter.verify_assets()
    rows = adapter.reconstructed_snar_context()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    assert hashlib.sha256(stream.getvalue().encode()).hexdigest() == adapter.digest(adapter.RAW / "snar.context.csv")
    catalog = json.loads((BASE / "catalog.json").read_text())
    sources = {source["id"]: source for source in catalog["sources"]}
    for table in catalog["tables"]:
        assert table_has_evidence(table, sources[table["source_id"]], catalog, ROOT)
    external = copy.deepcopy(sources["SNArSolution2021"])
    external["rights"]["scope"] = "external_repository"
    assert not source_has_evidence(external, catalog, ROOT)
