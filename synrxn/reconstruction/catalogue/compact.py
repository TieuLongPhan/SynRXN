"""Lossless internal views for audited overlaps in portable snapshots."""

from dataclasses import dataclass, field

CLASSIFICATION_PAIRS = ("schneider", "tpl", "uspto_50k")
NATCOMM_MATCH = "a.ground_truth = p.ground_truth"


@dataclass(frozen=True)
class Projection:
    """Columns recovered from one or more existing source relations."""

    child: str
    parents: tuple[str, ...]
    condition: str
    columns: dict[str, str]
    retained: dict[str, str] = field(default_factory=dict)


def projections():
    """Describe join keys and values that have been reconciled across sources."""
    for source in CLASSIFICATION_PAIRS:
        yield Projection(
            f"classification/{source}_u", (f"classification/{source}_b",),
            "a.r_id = p.r_id", {"label": "p.label", "split": "p.split"},
        )
    yield Projection(
        "aam/natcomm", ("aam/golden",), NATCOMM_MATCH,
        {field: f"p.{field}" for field in ("reactions", "rxn_mapper", "graphormer", "local_mapper")},
    )
    yield Projection(
        "classification/ehreact_activity", ("aam/ehreact",), "a.reaction_r_id = p.r_id",
        {field: f"p.{field}" for field in ("ground_truth", "rxn", "system", "input_type")},
    )
    yield Projection(
        "knowledge/MinotScope2026/scope_campaigns",
        tuple(f"C02/minot_{part}_data" for part in ("al1", "al2", "al3", "preal")),
        "a.source_record_id = p.record_id",
        {
            "inputs_json": "p.inputs_json",
            "targets_json": """CAST(json_object(
                'conversion_fraction', json_extract(p.targets_json, '$.sum_mono_bo_conv'),
                'native_hit_label', json_extract(p.targets_json, '$.binary')
            ) AS VARCHAR)""",
        },
    )
    yield Projection(
        "solution_context/snar_solution_context", ("property/snar",),
        "a.parent_record_id = 'property/snar:' || p.r_id",
        {"reaction_smiles": "p.rxn", "activation_free_energy_kcal_mol": "p.ea"},
    )
    yield Projection(
        "solution_context/cycloadd_solution_context", ("property/cycloadd",),
        "a.parent_record_id = 'property/cycloadd:' || p.r_id",
        {"reaction_smiles": "p.aam", "activation_free_energy_kcal_mol": "p.G_act",
         "reaction_free_energy_kcal_mol": "p.G_r", "source_split": "p.split"},
    )
    yield Projection(
        "selectivity/hitea_chiral_resolution", ("yield/hitea",),
        "a.source_record_id = p.source_record_id", {"selectivity_value": "p.selectivity"},
    )
    yield Projection(
        "bio/enzeng_inverse_biocatalyst_design", ("bio/enzeng_mutation_effects",),
        "a.source_member = p.source_member AND a.native_csv_row_index = p.native_csv_row_index",
        {field: f"p.{field}" for field in (
            "parent_sequence", "variant_sequence", "reaction_smiles",
            "reference_doi", "parent_sequence_group", "reaction_group",
        )},
    )
    yield Projection(
        "C03/cn_screen", ("condition_sharealike/ord_hte_condition",),
        """json_extract_string(a.context_json, '$.ord_reaction_id') = p.source_record_id
           AND json_extract_string(a.context_json, '$.dataset_id') = p.source_dataset_id""",
        {"targets_json": """CAST(json_object(
            'assay_yield_pct', CAST(p.reported_yield AS DOUBLE),
            'yield_measurements', json_extract(a.targets_json, '$.yield_measurements')
        ) AS VARCHAR)"""},
        {"targets_json": "json_object('yield_measurements', json_extract(targets_json, '$.yield_measurements'))"},
    )
    yield Projection(
        "knowledge/EnzChemRED2024/curation", ("knowledge/EnzChemRED2024/documents",),
        "a.parent_group = p.parent_group",
        {
            "inputs_json": """CAST(json_object('candidate_record', json_extract(a.inputs_json, '$.candidate_record'),
                                                'passages', json_extract(p.inputs_json, '$.passages')) AS VARCHAR)""",
            "targets_json": """CAST(json_object('corrected_record', p.targets_json::JSON,
                                                'needs_correction', a.targets_json::JSON) AS VARCHAR)""",
        },
        {"inputs_json": "json_object('candidate_record', json_extract(inputs_json, '$.candidate_record'))",
         "targets_json": "json_extract(targets_json, '$.needs_correction')"},
    )
    yield Projection(
        "physics/transition1x_final_ts_geometry", ("physics/transition1x_reactive_trajectories",),
        "a.trajectory_id = p.trajectory_id",
        {field: f"p.{field}" for field in (
            "formula", "native_split", "native_file", "native_group", "atomic_numbers",
            "theory", "charge", "multiplicity", "electronic_context_status",
        )},
    )


def same_rows(connection, original, reconstructed):
    """Compare both multisets, including null values and repeated rows."""
    return connection.execute(f"""
        WITH rebuilt AS ({reconstructed})
        SELECT NOT EXISTS (FROM {original} EXCEPT ALL FROM rebuilt)
           AND NOT EXISTS (FROM rebuilt EXCEPT ALL FROM {original})
    """).fetchone()[0]


def install(connection, mapping):
    """Replace verified duplicate columns with self-contained database views.

    Every view must reproduce the input SQL rows exactly. In particular, compact
    storage keeps original source metadata and numeric spelling when matched
    observations carry distinct fields.
    """
    relations = {record["table_key"]: record for record in mapping if record["status"] == "materialized_table"}
    connection.execute("CREATE SCHEMA source_extensions")
    for recipe in projections():
        required = (recipe.child, *recipe.parents)
        if any(key not in relations for key in required):
            continue
        record = relations[recipe.child]
        name = record["view"]
        extension = name.replace("data.", "source_extensions.", 1)
        parents = " UNION ALL ".join(f"FROM {relations[key]['view']}" for key in recipe.parents)
        joined = f"JOIN ({parents}) p ON {recipe.condition}"
        selected = select_columns(record, recipe.columns)
        kept = ["* EXCLUDE (" + ", ".join(recipe.columns) + ")"]
        kept.extend(f"{value}::VARCHAR AS {key}" for key, value in recipe.retained.items())
        connection.execute(f"CREATE TABLE {extension} AS SELECT {', '.join(kept)} FROM {name}")
        replace_relation(connection, record, f"SELECT {selected} FROM {extension} a {joined}")
    if "C06/single_solvent" in relations and "C06/catechol" in relations:
        compact_catechol(connection, relations["C06/single_solvent"], relations["C06/catechol"]["view"])
    for source in ("saccharose_hydrolysis", "pmg_284", "pmg_287"):
        parent = relations.get(f"process/nmr_{source}_snapshots")
        forecast = relations.get(f"process/nmr_{source}_progress")
        if parent and forecast:
            replace_relation(connection, forecast, forecast_query(parent["view"]))


def select_columns(record, replacements):
    """Render a projection in the published order."""
    fields = []
    for column in record["columns"]:
        name = column["name"]
        qualified = f'a."{name}"'
        fields.append(f'{replacements.get(name, qualified)} AS "{name}"')
    return ", ".join(fields)


def catechol_match_condition():
    """Match the published subset by experiment, conditions and measured fractions."""
    fields = (("context_json", "experiment"), ("inputs_json", "substrate_smiles"),
              ("inputs_json", "temperature_c"), ("inputs_json", "residence_time_min"))
    conditions = [f"(a.{field}->>'{key}') = (p.{field}->>'{key}')" for field, key in fields]
    conditions.extend(
        f"abs((a.targets_json->>'{key}')::DOUBLE - (p.targets_json->>'{key}')::DOUBLE) < 1e-10"
        for key in ("SM", "Product 2", "Product 3")
    )
    return " AND ".join(conditions)


def compact_catechol(connection, record, parent):
    """Keep the subset's condition labels and exact response spellings."""
    child = record["view"]
    extension = child.replace("data.", "source_extensions.", 1)
    connection.execute(f"""
        CREATE TABLE {extension} AS
        SELECT a.* EXCLUDE (targets_json), p.record_id AS full_panel_record_id,
               CASE WHEN a.targets_json <> p.targets_json THEN a.targets_json END AS source_response_text
        FROM {child} a JOIN {parent} p ON {catechol_match_condition()}
    """)
    selected = select_columns(record, {"targets_json": "coalesce(a.source_response_text, p.targets_json)"})
    replace_relation(connection, record, f"""
        SELECT {selected} FROM {extension} a JOIN {parent} p ON a.full_panel_record_id = p.record_id
    """)


def replace_relation(connection, record, query):
    """Validate an internal projection before substituting it for an imported table."""
    name = record["view"]
    rebuilt_schema = [{"name": col[0], "type": col[1]} for col in connection.execute(
        f"DESCRIBE {query}"
    ).fetchall()]
    if record["columns"] != rebuilt_schema or not same_rows(connection, name, query):
        raise ValueError(f"Compact view failed to reproduce source rows: {record['table_key']}")
    connection.execute(f"DROP TABLE {name}")
    connection.execute(f"CREATE VIEW {name} AS {query}")
    connection.execute("""UPDATE catalog.dataset_tables SET view_status = 'internal_view'
                          WHERE namespace = ? AND table_key = ?""",
                       [record["namespace"], record["table_key"]])
    record.update(status="internal_view", sql=query)


def forecast_query(parent):
    """Pair each snapshot with its adjacent reference without storing it twice."""
    return rf"""
        WITH ordered AS (
            SELECT payload AS current,
                   lead(payload) OVER (
                      PARTITION BY payload->>'trajectory_id'
                      ORDER BY (payload->>'native_spectrum_column')::BIGINT
                   ) AS following
            FROM {parent}
        )
        SELECT replace(json_object(
            'component_names', current->'component_names',
            'elapsed_seconds', current->'elapsed_seconds',
            'integration_windows_ppm', current->'integration_windows_ppm',
            'next_reference_area_fractions', following->'reference_area_fractions',
            'record_id', (current->>'record_id') || ':next',
            'spectrum_column', current->'spectrum_column',
            'spectrum_path', current->'spectrum_path',
            'target_elapsed_seconds', following->'elapsed_seconds',
            'target_native_spectrum_column', following->'native_spectrum_column',
            'trajectory_id', current->'trajectory_id'
        )::VARCHAR,
           '"next_reference_area_fractions":' || (following->'reference_area_fractions')::VARCHAR,
           '"next_reference_area_fractions":' ||
               regexp_extract(following::VARCHAR, '"reference_area_fractions":(\[[^\]]*\])', 1)
        )::JSON AS payload FROM ordered WHERE following IS NOT NULL
    """
