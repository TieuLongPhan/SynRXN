"""Normalize the Minot borylation experiments."""

import re
from pathlib import Path

from synrxn.reconstruction.expansion.storage import observation

from synrxn.reconstruction.families._observations import csv_rows, selected


def borylation(ctx):
    for name in ["Minot_preAL_Data.csv", "Minot_AL1_Data.csv", "Minot_AL2_Data.csv", "Minot_AL3_Data.csv"]:
        if not ctx.has(name):
            continue
        table = Path(name).stem.lower()
        input_fields = []

        def rows(name=name, table=table):
            for i, row in csv_rows(ctx.path(name), ";" if "preAL" in name else "\t"):
                keys = [k for k in row if re.search(r"_(?:smiles|eq|fraction)$", k)
                        or k in {"temperature_deg_c", "time_h", "atmosphere", "scale_mol", "concentration_mol_l"}]
                input_fields[:] = sorted(set(input_fields + keys))
                inputs = selected(row, keys)
                targets = selected(row, ["sum_mono_bo_conv", "binary"])
                # Substrate groups stay together across all adaptive rounds.
                parent = str(row.get("startingmat_1_smiles", ""))
                yield observation("C02", table, i, parent, inputs=inputs, targets=targets,
                                  native=row, context={"campaign": name, "source_record_id": row.get("rxn_id"),
                                                       "assay": row.get("conversion_determination"),
                                                       "source_binary_threshold": "not re-derived"})
        result = ctx.emit(table, rows(), unit="HTE observation", assets=[name], tasks={},
                          units={"sum_mono_bo_conv": "fraction", "binary": "source-defined binary label"},
                          qualification="Conversion is not isolated yield. Native binary labels are preserved; the adapter does not infer an undocumented threshold.")
        result["task_fields"] = {
            "yield_conversion": {"inputs": input_fields, "targets": ["sum_mono_bo_conv"]},
            "reaction_success": {"inputs": input_fields, "targets": ["binary"]},
            "condition_ranking": {"inputs": input_fields, "targets": ["sum_mono_bo_conv"]},
            "sequential_optimization": {"inputs": input_fields, "targets": ["sum_mono_bo_conv"]}}
        result["tasks"] = list(result["task_fields"])
    name = "Regio_Data.csv"
    if ctx.has(name):
        def rows():
            for i, row in csv_rows(ctx.path(name), "\t"):
                yield observation("C02", "regio", i, str(row["SMILES"]),
                                  inputs={"substrate_smiles": row["SMILES"]},
                                  targets={"product_smiles": row["product_smiles"]}, native=row,
                                  evidence="reported_product_annotation")
        ctx.emit("regio", rows(), unit="substrate-product annotation", assets=[name], tasks={
            "regioselectivity": {"inputs": ["substrate_smiles"], "targets": ["product_smiles"]}},
            qualification="Product annotations do not establish full product ratios or symmetry-aware reactive-site labels.")

