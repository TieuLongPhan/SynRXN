"""Normalize the OrgAIcat enantioselectivity workbooks."""

import re
from pathlib import Path

from synrxn.reconstruction.expansion.storage import clean, observation

from synrxn.reconstruction.families._observations import meaningful, selected


def orgaicat(ctx):
    import pandas as pd

    for name in sorted(ctx.assets):
        if not name.endswith(".xlsx") or not ctx.has(name):
            continue
        table = Path(name).stem.replace("-", "_").lower()
        frame = pd.read_excel(ctx.path(name), sheet_name=0)
        # Other workbook sheets are retained in the immutable source workbook;
        # they are not counted as additional measurements.
        input_fields = [k for k in frame.columns if re.search(r"smiles|temperature", str(k), re.I)]
        target_fields = [k for k in ["abs_ee", "p1_ee", "experimental_deltaG"] if k in frame]
        if not target_fields:
            ctx.issue(name, "native Output endpoint has no documented unit; retained as source, excluded from ee views")
            continue

        def rows(frame=frame, table=table, input_fields=input_fields, target_fields=target_fields):
            for i, values in enumerate(frame.to_dict("records")):
                row = clean(values)
                doi = row.get("doi", row.get("DOI"))
                # Source papers group literature observations; unpublished
                # validation campaigns group by workbook, never individual rows.
                parent = "paper:" + str(doi) if meaningful(doi) else "campaign:" + table
                yield observation("C16", table, i, parent, inputs=selected(row, input_fields),
                                  targets=selected(row, target_fields), native=row,
                                  context={"doi": doi, "campaign": table,
                                           "orientation": "absolute" if "abs_ee" in row else "source-defined",
                                           "deltaG_evidence": "ratio-derived, not a measured barrier"})
        ctx.emit(table, rows(), unit="literature or validation observation", assets=[name], tasks={
            "enantioselectivity": {"inputs": input_fields, "targets": target_fields}},
            units={k: "percent" if k in {"abs_ee", "p1_ee"} else "kcal/mol (source-derived)" for k in target_fields},
            qualification="Absolute and source-defined ee remain distinct endpoints. Predictions present in validation workbooks are retained only in native provenance and cannot enter task inputs.")

