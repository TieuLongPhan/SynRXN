"""Normalize the catechol and Claisen flow panels."""

from synrxn.reconstruction.expansion.storage import encoded, observation

from synrxn.reconstruction.families._observations import csv_rows, selected


def catechol(ctx):
    specifications = [("catechol", "data/full_data/catechol_full_data_yields.csv", "primary"),
                      ("claisen", "data/full_data/claisen_data_clean.csv", "primary"),
                      ("single_solvent", "data/single_solvent/catechol_single_solvent_yields.csv", "overlapping_view")]
    for table, name, role in specifications:
        if not ctx.has(name):
            continue

        def rows(table=table, name=name):
            for i, row in csv_rows(ctx.path(name)):
                inputs = {"substrate_smiles": row["SM SMILES"], "temperature_c": row["Temperature"],
                          "residence_time_min": row["Residence Time"]}
                inputs.update(selected(row, ["SOLVENT A SMILES", "SOLVENT B SMILES", "SolventB%", "SOLVENT SMILES"]))
                targets = selected(row, ["SM", "Product", "Product 2", "Product 3"])
                solvent_pair = [row.get("SOLVENT A SMILES", row.get("SOLVENT SMILES")), row.get("SOLVENT B SMILES")]
                # EXP NUM is a source row ID, not a campaign identifier.
                parent = encoded([row["SM SMILES"], solvent_pair])
                yield observation("C06", table, i, parent, inputs=inputs, targets=targets, native=row,
                                  context={"experiment": row["EXP NUM"], "ramp": row.get("RAMP NUM"),
                                           "product_identifiers": selected(row, ["Product SMILES", "Product 2 SMILES", "Product 3 SMILES"]),
                                           "unit_note": "native yield fractions; no simplex renormalization"})
        targets = ["SM", "Product 2", "Product 3"] if table != "claisen" else ["SM", "Product"]
        fields = ["substrate_smiles", "temperature_c", "residence_time_min", "SOLVENT A SMILES", "SOLVENT B SMILES", "SolventB%", "SOLVENT SMILES"]
        ctx.emit(table, rows(), unit="flow-condition observation", assets=[name], role=role, tasks={
            "product_distribution": {"inputs": fields, "targets": targets},
            "sequential_optimization": {"inputs": fields, "targets": targets}},
            units={k: "fraction" for k in targets},
            qualification="Single-solvent observations overlap the complete panel. Residence time is an input, not an observed time trajectory. Missing mass is not renormalized.")

