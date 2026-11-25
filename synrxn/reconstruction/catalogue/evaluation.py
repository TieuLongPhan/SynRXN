"""Source-qualified evaluation views over the stored catalogue."""

from synrxn.reconstruction.catalogue.compact import (
    CLASSIFICATION_PAIRS,
    NATCOMM_MATCH,
    catechol_match_condition,
)

GROUPED_SPLITS = {
    "classification/ecreact": ("rxn", "split", "split || ':' || r_id"),
    "property/rdb7": ("aam", "split", "r_id"),
    "property/rgd1": ("aam", "split", "r_id"),
    "synthesis/uspto_50k": ("aam", "split", "r_id"),
    "synthesis/uspto_mit": ("aam", "split", "r_id"),
    "synthesis/uspto_500": ("rxn", "split", "r_id"),
}


def install(connection, mapping):
    """Expose scoring populations and literal-reaction splits with source IDs."""
    relations = {record["table_key"]: record["view"] for record in mapping
                 if record["status"] in {"file_view", "materialized_table", "internal_view"}}
    if not relations:
        return
    connection.execute("CREATE SCHEMA evaluation")
    for key, (reaction, source_split, record_id) in GROUPED_SPLITS.items():
        if key not in relations:
            continue
        name = key.replace("/", "_")
        connection.execute(f"""
            CREATE VIEW evaluation.{name}_split AS
            WITH groups AS (
                SELECT {record_id} AS source_record_id, {source_split} AS source_split,
                       sha256('synrxn-audit-v1:{key}:' || {reaction}) AS group_id
                FROM {relations[key]}
            ), assignments AS (
                SELECT *, cast('0x' || left(group_id, 8) AS UBIGINT) % 100 AS bucket
                FROM groups
            )
            SELECT source_record_id, source_split, group_id,
                   CASE WHEN bucket < 80 THEN 'train'
                        WHEN bucket < 90 THEN 'validation'
                        WHEN bucket >= 90 THEN 'test' END AS proposed_split
            FROM assignments
        """)
    for key, relation in relations.items():
        if key.startswith("rbl/"):
            name = key.replace("/", "_")
            connection.execute(f"""
                CREATE VIEW evaluation.{name}_references AS
                SELECT * FROM {relation}
                WHERE nullif(trim(ground_truth), '') IS NOT NULL
            """)
    if "property/phosphatase" in relations:
        connection.execute(f"""
            CREATE VIEW evaluation.phosphatase_assay_response AS
            SELECT * RENAME ("Conversion" AS assay_response)
            FROM {relations["property/phosphatase"]}
        """)
    install_overlap_views(connection, relations)


def install_overlap_views(connection, relations):
    """Expose preferred inputs, subset membership and paired classification splits."""
    if snar := relations.get("solution_context/snar_solution_context"):
        connection.execute(f"""
            CREATE VIEW evaluation.snar AS SELECT * FROM {snar};
            CREATE VIEW evaluation.snar_reaction_only AS
            SELECT source_record_id, reaction_smiles, activation_free_energy_kcal_mol,
                   reaction_group, proposed_split, source_split
            FROM evaluation.snar;
        """)
    if cycloaddition := relations.get("solution_context/cycloadd_solution_context"):
        connection.execute(f"""
            CREATE VIEW evaluation.cycloaddition AS SELECT * FROM {cycloaddition};
            CREATE VIEW evaluation.cycloaddition_activation_barrier AS
            SELECT * EXCLUDE (reaction_free_energy_kcal_mol) FROM evaluation.cycloaddition;
            CREATE VIEW evaluation.cycloaddition_reaction_energetics AS
            SELECT * EXCLUDE (activation_free_energy_kcal_mol) FROM evaluation.cycloaddition;
        """)
    if (catechol := relations.get("C06/catechol")) and (single := relations.get("C06/single_solvent")):
        connection.execute(f"""
            CREATE VIEW evaluation.catechol_single_solvent AS
            SELECT a.*, p.record_id AS full_panel_record_id, p.split AS full_panel_split
            FROM {single} a JOIN {catechol} p ON {catechol_match_condition()};
            CREATE VIEW evaluation.catechol AS
            SELECT p.*, EXISTS (
                SELECT 1 FROM evaluation.catechol_single_solvent a
                WHERE a.full_panel_record_id = p.record_id
            ) AS in_single_solvent_subset
            FROM {catechol} p;
        """)
    if (golden := relations.get("aam/golden")) and (natcomm := relations.get("aam/natcomm")):
        connection.execute(f"""
            CREATE VIEW evaluation.atom_mapping_natcomm AS
            SELECT a.*, p.r_id AS golden_record_id
            FROM {natcomm} a JOIN {golden} p ON {NATCOMM_MATCH};
            CREATE VIEW evaluation.atom_mapping_golden_only AS
            SELECT g.* FROM {golden} g
            WHERE NOT EXISTS (
                SELECT 1 FROM evaluation.atom_mapping_natcomm n WHERE n.golden_record_id = g.r_id
            );
        """)
    for source in CLASSIFICATION_PAIRS:
        balanced = relations.get(f"classification/{source}_b")
        unbalanced = relations.get(f"classification/{source}_u")
        if balanced and unbalanced:
            connection.execute(f"""
                CREATE VIEW evaluation.classification_{source}_paired_split AS
                SELECT b.r_id AS source_record_id, b.split AS source_split
                FROM {balanced} b JOIN {unbalanced} u USING (r_id)
                WHERE b.split IS NOT DISTINCT FROM u.split
            """)
