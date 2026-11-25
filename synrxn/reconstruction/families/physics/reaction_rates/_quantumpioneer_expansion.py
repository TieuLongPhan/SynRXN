"""Normalize the QuantumPioneer kinetics and transition-state files."""

import re

from synrxn.reconstruction.expansion.storage import observation

from synrxn.reconstruction.families._observations import csv_rows, selected


def undirected_reaction(smiles: str) -> str:
    """Group forward/reverse and atom-map order variants conservatively.

    This is a string grouping helper, not chemical deduplication or mapping.
    """
    halves = str(smiles).split(">")
    if len(halves) != 3:
        return str(smiles)
    sides = [".".join(sorted(re.sub(r":\d+(?=\])", "", h).split(".")))
             for h in (halves[0], halves[-1])]
    return "<>".join(sorted(sides))


def quantumpioneer(ctx):
    for name, table in [("quantumpioneer_kinetics_dataset.csv", "kinetics"),
                        ("quantumpioneer_thermo_dataset_transition_states.csv", "ts_thermo")]:
        if not ctx.has(name):
            continue
        fields = (["k_298", "A_low", "Ea_low", "A_high", "Ea_high", "barrier", "Hrxn", "deltaHrxn298", "deltaGrxn298", "P2M"]
                  if table == "kinetics" else ["dlpno_sp_hartree", "dft_zpe_scaled_hartree"])

        def rows(name=name, table=table, fields=fields):
            for i, row in csv_rows(ctx.path(name)):
                rxn = row["rxn_smi"]
                yield observation("C11", table, i, undirected_reaction(rxn), inputs={"rxn_smi": rxn},
                                  targets=selected(row, fields), native=row, evidence="quantum_calculation",
                                  context={"temperature_k": 298, "arrhenius_low_range_k": [300, 1000],
                                           "arrhenius_high_range_k": [1000, 2000]})
        ctx.emit(table, rows(), unit="computed reaction endpoint", assets=[name], tasks={
            "rate_constants": {"inputs": ["rxn_smi"], "targets": fields}} if table == "kinetics" else {},
            units={k: "m^3/(mol s)" if k in {"k_298", "A_low", "A_high"} else "Hartree" if "hartree" in k else "kcal/mol" for k in fields},
            qualification="Rate coefficients are calculated bimolecular quantities at the stated temperatures; they are not enzyme kcat or experimentally measured kinetic traces.")

