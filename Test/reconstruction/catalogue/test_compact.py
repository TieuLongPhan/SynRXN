"""Regression checks for audited projections and scoring views."""

import duckdb
import pytest

from synrxn.reconstruction.catalogue import compact, evaluation


def classification_fixture(connection):
    """Register two linked classification variants."""
    connection.execute("""
        CREATE SCHEMA catalog;
        CREATE SCHEMA data;
        CREATE TABLE catalog.dataset_tables (namespace VARCHAR, table_key VARCHAR, view_status VARCHAR);
        CREATE TABLE data.schneider_b AS
            SELECT * FROM (VALUES ('01', 'balanced', 'label', 'test')) t(r_id, rxn, label, split);
        CREATE TABLE data.schneider_u AS
            SELECT * FROM (VALUES ('01', 'unbalanced', 'label', 'test')) t(r_id, rxn, label, split);
    """)
    return [
        {
            "namespace": "canonical", "table_key": f"classification/schneider_{kind}",
            "status": "materialized_table", "view": f"data.schneider_{kind}",
            "columns": [{"name": field, "type": "VARCHAR"} for field in ("r_id", "rxn", "label", "split")],
        }
        for kind in ("b", "u")
    ]


def test_compact_projection_reuses_fields_and_preserves_rows():
    """Retain the source-specific input under the original relation name."""
    with duckdb.connect() as con:
        mapping = classification_fixture(con)
        compact.install(con, mapping)
        assert mapping[1]["status"] == "internal_view"
        assert con.execute("FROM data.schneider_u").fetchall() == [("01", "unbalanced", "label", "test")]
        assert con.execute("FROM source_extensions.schneider_u").fetchall() == [("01", "unbalanced")]


def test_compact_projection_rejects_different_labels():
    """Refuse to substitute a compact view if it would change the source."""
    with duckdb.connect() as con:
        mapping = classification_fixture(con)
        con.execute("UPDATE data.schneider_u SET label = 'other'")
        with pytest.raises(ValueError, match="failed to reproduce source rows"):
            compact.install(con, mapping)
        assert con.execute("SELECT label FROM data.schneider_u").fetchall() == [("other",)]


def test_evaluation_views_keep_source_split_and_drop_missing_references():
    """Repeated reactions share a proposed fold and a missing answer stays out of scoring."""
    with duckdb.connect() as con:
        con.execute("""
            CREATE SCHEMA data;
            CREATE TABLE data.chemistry AS
                SELECT * FROM (VALUES ('1', 'train', 'A>>B'), ('1', 'test', 'A>>B')) s(r_id, split, rxn);
            CREATE TABLE data.balancing AS
                SELECT * FROM (VALUES ('with-answer', 'A>>B'), ('missing', NULL), ('empty', ' ')) s(r_id, ground_truth);
        """)
        evaluation.install(con, [
            {"table_key": "classification/ecreact", "view": "data.chemistry", "status": "materialized_table"},
            {"table_key": "rbl/complex", "view": "data.balancing", "status": "materialized_table"},
        ])
        rows = con.execute("""SELECT source_record_id, source_split, proposed_split
                              FROM evaluation.classification_ecreact_split ORDER BY source_split""").fetchall()
        assert [(row[0], row[1]) for row in rows] == [("test:1", "test"), ("train:1", "train")]
        assert rows[0][2] in {"train", "validation", "test"}
        assert rows[0][2] == rows[1][2]
        assert con.execute("FROM evaluation.rbl_complex_references").fetchall() == [("with-answer", "A>>B")]
        assert con.execute("SELECT count(*) FROM data.balancing").fetchone()[0] == 3


def test_paired_classification_and_catechol_publish_comparable_splits():
    with duckdb.connect() as con:
        mapping = classification_fixture(con)
        con.execute("""
            CREATE TABLE data.catechol AS
                SELECT 'panel-1' AS record_id, 'train' AS split,
                       '{"experiment":"exp-1"}' AS context_json,
                       '{"substrate_smiles":"A","temperature_c":90,"residence_time_min":5,"minor_solvent":0.01}' AS inputs_json,
                       '{"SM":0.25,"Product 2":0.5,"Product 3":0.25}' AS targets_json;
            CREATE TABLE data.single_solvent AS
                SELECT 'subset-1' AS record_id, 'test' AS split, context_json,
                       '{"substrate_smiles":"A","temperature_c":90,"residence_time_min":5}' AS inputs_json,
                       targets_json
                FROM data.catechol;
        """)
        mapping.extend([
            {"table_key": "C06/catechol", "view": "data.catechol", "status": "materialized_table"},
            {"table_key": "C06/single_solvent", "view": "data.single_solvent", "status": "materialized_table"},
        ])
        evaluation.install(con, mapping)
        assert con.execute("FROM evaluation.classification_schneider_paired_split").fetchall() == [("01", "test")]
        assert con.execute("""
            SELECT record_id, split, full_panel_record_id, full_panel_split, inputs_json->>'minor_solvent'
            FROM evaluation.catechol_single_solvent
        """).fetchall() == [("subset-1", "test", "panel-1", "train", None)]
        assert con.execute("""
            SELECT record_id, in_single_solvent_subset, inputs_json->>'minor_solvent'
            FROM evaluation.catechol
        """).fetchall() == [("panel-1", True, "0.01")]


def test_golden_only_and_natcomm_keep_separate_benchmark_populations():
    with duckdb.connect() as con:
        con.execute("""
            CREATE SCHEMA data;
            CREATE TABLE data.golden AS
                SELECT * FROM (VALUES ('g1', 'shared', 'archived'), ('g2', 'other', 'result'))
                t(r_id, ground_truth, rdt);
            CREATE TABLE data.natcomm AS
                SELECT 'n1' AS r_id, 'shared' AS ground_truth, 'published' AS rdt;
        """)
        evaluation.install(con, [
            {"table_key": "aam/golden", "view": "data.golden", "status": "materialized_table"},
            {"table_key": "aam/natcomm", "view": "data.natcomm", "status": "materialized_table"},
        ])
        assert con.execute("SELECT r_id FROM evaluation.atom_mapping_golden_only").fetchall() == [("g2",)]
        assert con.execute("""
            SELECT r_id, golden_record_id, rdt FROM evaluation.atom_mapping_natcomm
        """).fetchall() == [("n1", "g1", "published")]
        assert con.execute("SELECT count(*) FROM data.golden").fetchone() == (2,)
