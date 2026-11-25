"""Normalize the RDB19 solvent calculations."""

import tarfile

from synrxn.reconstruction.expansion.storage import observation

from synrxn.reconstruction.families._observations import csv_rows, meaningful, selected


def rdb19(ctx):
    name = "Datasets.tar.bz2"
    if not ctx.has(name):
        return
    with tarfile.open(ctx.path(name), "r:bz2") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".csv")]
        primary = [m for m in members if "all_rxn_all_solvent_bidirection_resonance.csv" in m.name]
        for member in primary:
            def rows():
                with archive.extractfile(member) as stream:
                    for i, row in csv_rows(stream):
                        targets = {k: v for k, v in row.items() if k.startswith("Delta") and meaningful(v)}
                        yield observation("C07", "solvent_panel", i, "rxn:" + str(row["rxn_idx"]),
                                          inputs=selected(row, ["rxn_smi", "solvent_smi", "solvent_name", "direction"]),
                                          targets=targets, native=row, evidence="quantum_calculation",
                                          context={"rxn_idx": row["rxn_idx"], "resonance_idx": row.get("resonance_idx"),
                                                   "Kmeans_split_cluster_idx": row.get("Kmeans_split_cluster_idx")})
            targets = ["DeltaG_ET_gas_molar_kcal_mol", "DeltaG_EP_gas_molar_kcal_mol",
                       "DeltaDeltaG_ET_solv_molar_kcal_mol", "DeltaDeltaG_EP_solv_molar_kcal_mol",
                       "DeltaG_ET_soln_molar_kcal_mol", "DeltaG_EP_soln_molar_kcal_mol"]
            ctx.emit("solvent_panel", rows(), unit="reaction-direction-resonance-solvent calculation", assets=[name], tasks={
                "solvent_effects": {"inputs": ["rxn_smi", "solvent_smi", "direction"], "targets": targets}},
                units={k: "kcal/mol" for k in targets},
                qualification="All solvents, directions and resonance variants share the source reaction parent. Smaller source subsets are not added as independent observations.")
        if not primary:
            raise ValueError("RDB19 primary CSV absent from archive")

