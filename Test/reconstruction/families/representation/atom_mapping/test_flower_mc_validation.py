"""Validate pathway-aware export, persisted MC assignments and resumed selection."""

import csv
import gzip
import json

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_mc_export as export
from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_mc_store as store
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import mechanistic_center, pathway_mtg

INITIAL = "[CH3:1][C:2](=[O:3])[Cl:4].[NH3:5]"
MIDDLE = "[CH3:1][C:2]([O-:3])([Cl:4])[NH3+:5]"
FINAL = "[CH3:1][C:2](=[O:3])[NH3+:5].[Cl-:4]"
REACTIONS = [INITIAL + ">>" + MIDDLE, MIDDLE + ">>" + FINAL]


def member(number, rc="legacy-same", steps=2):
    return {
        "membership_id": f"path-{number}",
        "n_steps": str(steps),
        "rc_cluster_id": rc,
        "reference_id": f"source-{number}",
        "source_group": str(number),
        "source_lines": "[1,2]",
    }


def test_active_transient_catalyst_is_kept_and_inactive_salt_removed():
    # Water/amine are restored after proton transfer, but remain pathway-active.
    first = "[O:1]([H:2])[H:3].[N:4]([H:5])([H:6])[H:7].[Na+:8]"
    middle = "[O-:1][H:3].[N+:4]([H:2])([H:5])([H:6])[H:7].[Na+:8]"
    reactions = [first + ">>" + middle, middle + ">>" + first]
    center, _ = mechanistic_center(pathway_mtg(reactions))
    endpoints = export.pathway_endpoints(reactions, set(center))
    truth, rxn, details = export.normalize_pathway(endpoints, set(center))
    assert ":8]" not in truth and ":2]" in truth
    assert all(f":{number}]" in truth for number in (1, 2, 4))
    assert details["explicit_hydrogen_maps_retained"] == [2]
    assert details["nonparticipating_hydrogens_made_implicit_per_side"] == 4
    assert all(side for side in rxn.split(">>"))


def test_original_mappings_and_endpoint_chemistry_survive_normalization():
    graph, _ = mechanistic_center(pathway_mtg(REACTIONS))
    active = export.pathway_endpoints(REACTIONS, set(graph))
    truth, _, info = export.normalize_pathway(active, set(graph))
    assert set(map(int, export.MAP.findall(truth.split(">>")[0]))) == {1, 2, 3, 4, 5}
    assert info["all_pathway_active_atoms_preserved"] and info["endpoint_chemistry_verified"]


def test_checkpoint_rejects_source_drift(tmp_path):
    path = tmp_path / "checkpoint.sqlite3"
    db = store.open_store(path, {"source": "first"})
    db.close()
    with pytest.raises(ValueError, match="Checkpoint source"):
        store.open_store(path, {"source": "changed"})


def test_resume_restores_exact_classes_and_mc_selection_ignores_legacy_rc(tmp_path):
    path = tmp_path / "checkpoint.sqlite3"
    config = {"source": "fixture"}
    records = [member(1, "rc-one"), member(2, "rc-two")]
    db = store.open_store(path, config)
    store.prepare_members(db, records)
    graph, stats = mechanistic_center(pathway_mtg(REACTIONS))
    active = export.pathway_endpoints(REACTIONS, set(graph))
    clusters = store.load_clusters(db)
    store.save_batch(db, clusters, [("path-1", graph, stats, active, None)])
    db.close()
    db = store.open_store(path, config)
    store.prepare_members(db, records)
    clusters = store.load_clusters(db)
    store.save_batch(db, clusters, [("path-2", graph, stats, active, None)])
    assert store.counts(db) == {"done": 2}
    assert db.execute("SELECT COUNT(*) FROM centers").fetchone()[0] == 1
    output = tmp_path / "output"
    first = export.export_bundle(db, output, [], 42, config)
    assert first["rows"] == 1 and first["center_type_counts"] == {"MC": 1}
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    export.export_bundle(db, output, [], 42, config)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    with gzip.open(output / "aam.csv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == list(export.COLUMNS)
    assert rows[0]["role"] == "val" and rows[0]["steps"] == "2"
    db.close()


def test_failed_pathways_remain_in_membership_metadata(tmp_path):
    db = store.open_store(tmp_path / "checkpoint.sqlite3", {"source": "fixture"})
    store.prepare_members(db, [member(1)])
    store.save_batch(db, store.load_clusters(db), [("path-1", None, None, None, "source failure")])
    output = tmp_path / "output"
    report = export.export_bundle(db, output, [], 42, {"source": "fixture"})
    assert report["failed_memberships"] == 1
    with gzip.open(output / "memberships.jsonl.gz", "rt") as handle:
        row = json.loads(next(handle))
    assert row["status"] == "failed" and row["error"] == "source failure"
    db.close()


def test_uncapped_prefixes_keep_shared_states_separate_and_stop_cycles():
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_pipeline import choose_paths
    edges = [(f'A{i}', f'A{i+1}', i + 1) for i in range(25)]
    paths, _ = choose_paths(edges, {'A25'}, 4)
    assert len(paths[0]) == 25
    shared = [('A', 'B', 1), ('B', 'C', 2), ('A', 'C', 3), ('C', 'D', 4)]
    paths, _ = choose_paths(shared, {'D'}, 4)
    assert [[edge[2] for edge in path] for path in paths] == [[1, 2, 4], [3, 4]]
    paths, stats = choose_paths([('A', 'B', 1), ('B', 'C', 2), ('C', 'B', 3)], set(), 4)
    assert len(paths[0]) == 2 and stats['stop_revisited_state'] == 1


def test_single_mtg_prefix_projection_matches_independent_prefix_construction():
    import networkx as nx
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import growing_centers
    reactions = REACTIONS + [FINAL + '>>' + MIDDLE, MIDDLE + '>>' + INITIAL]
    for length, graph, statistics in growing_centers(reactions):
        expected, details = mechanistic_center(pathway_mtg(reactions[:length]))
        assert nx.utils.graphs_equal(graph, expected)
        assert statistics == details


def test_disk_mc_classes_survive_resume_with_bounded_cache(tmp_path):
    db = store.open_store(tmp_path / 'classes.sqlite3', {'source': 'fixture'})
    first, _ = mechanistic_center(pathway_mtg(REACTIONS))
    assert store.DiskClusters(db, capacity=1).assign(first) == 0
    db.commit()
    assert store.DiskClusters(db, capacity=1).assign(first) == 0
    assert db.execute('SELECT COUNT(*) FROM centers').fetchone()[0] == 1
    db.close()
