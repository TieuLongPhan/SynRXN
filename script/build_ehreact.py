"""Build SynRXN datasets from the EHreact manuscript assay tables.

Upstream repository:
    https://github.com/hesther/ehreact

Documentation:
    https://hesther.github.io/ehreact/

Paper:
    https://doi.org/10.1021/acs.jcim.1c00921

License:
    BSD 3-Clause (the upstream repository-level LICENSE).

The EHreact repository publishes nine binarized assay tables used in the
manuscript.  Every source row contains an atom-mapped reaction SMILES (``AAM``)
and one or more binary assay columns.  This builder creates two complementary
outputs:

``Data/aam/ehreact.csv.gz``
    One row per upstream reaction, preserving all assay labels as JSON.

``Data/classification/ehreact_activity.csv.gz``
    Long-form table with one row per reaction/assay pair and a binary
    ``activity`` target.

The manuscript reports that reactions were mapped with RDT, anomalous reaction
centres were flagged, and mappings were manually corrected.  This is recorded
as dataset-level/manual mapping curation.  EHreact does not publish a row-level
independent-validation field, so ``validated`` is deliberately left null.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import requests


REPOSITORY = "https://github.com/hesther/ehreact"
RAW_BASE = "https://raw.githubusercontent.com/hesther/ehreact"
DEFAULT_REF = "e6d29c21bd24ee23302e5e441680af4f6ed1d458"

TABLES = (
    "alcoholdehydrogenase",
    "amidinotransferase",
    "aminedehydrogenase",
    "carboxylmethyltransferase",
    "csp2_csp3_coupling",
    "dehalogenase",
    "nitrilase",
    "transaminase",
    "tryptophan_synthase",
)

DEFAULT_OUTPUT = Path("Data/aam/ehreact.csv.gz")
DEFAULT_ASSAY_OUTPUT = Path("Data/classification/ehreact_activity.csv.gz")

REACTION_COLUMNS = [
    "r_id",
    "ground_truth",
    "rxn",
    "original_id",
    "system",
    "input",
    "input_type",
    "assay_labels",
    "n_assays",
    "n_active",
    "mapping_curated",
    "validated",
]

ASSAY_COLUMNS = [
    "r_id",
    "reaction_r_id",
    "ground_truth",
    "rxn",
    "original_id",
    "system",
    "input",
    "input_type",
    "assay",
    "activity",
    "mapping_curated",
    "validated",
]


def looks_mapped(reaction_smiles: str) -> bool:
    """Return whether a reaction SMILES contains at least one atom map."""
    return bool(re.search(r":\d+(?=[^\]]*\])", str(reaction_smiles)))


def remove_atom_mapping_fallback(reaction_smiles: str) -> str:
    """Remove bracket atom-map numbers without changing the remaining text."""
    return re.sub(r":\d+(?=[^\]]*\])", "", str(reaction_smiles))


def remove_atom_mapping(reaction_smiles: str) -> str:
    """Remove AAM with SynKit, falling back for unsupported pseudo-atoms."""
    try:
        from synkit.Chem.Reaction.standardize import Standardize

        unmapped = Standardize().fit(
            str(reaction_smiles),
            remove_aam=True,
            ignore_stereo=False,
        )
        if unmapped and not looks_mapped(unmapped):
            return unmapped
    except Exception:
        pass

    return remove_atom_mapping_fallback(reaction_smiles)


def _read_source_bytes(
    name: str,
    *,
    ref: str,
    input_dir: Path | None,
    timeout: float,
) -> tuple[bytes, str]:
    filename = f"{name}.csv"
    if input_dir is not None:
        path = input_dir / filename
        return path.read_bytes(), str(path.resolve())

    url = f"{RAW_BASE}/{ref}/data_manuscript/{filename}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content, url


def load_tables(
    *,
    ref: str = DEFAULT_REF,
    input_dir: Path | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Load all nine EHreact manuscript CSVs and capture source hashes."""
    tables: dict[str, pd.DataFrame] = {}
    sources: list[dict[str, Any]] = []

    for name in TABLES:
        content, location = _read_source_bytes(
            name,
            ref=ref,
            input_dir=input_dir,
            timeout=timeout,
        )
        frame = pd.read_csv(io.BytesIO(content))
        tables[name] = frame
        sources.append(
            {
                "system": name,
                "location": location,
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": len(frame),
                "columns": list(frame.columns),
            }
        )
        logging.info("Loaded %-28s %4d rows", name, len(frame))

    return tables, sources


def _source_schema(
    frame: pd.DataFrame,
    system: str,
) -> tuple[str, str, list[str]]:
    required = {"AAM", "Unnamed: 0"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{system}: missing required columns {sorted(missing)}"
        )

    if "Substrate" in frame.columns:
        input_column = "Substrate"
        input_type = "substrate"
    elif "Transformation" in frame.columns:
        input_column = "Transformation"
        input_type = "transformation"
    else:
        raise ValueError(f"{system}: missing Substrate/Transformation column")

    metadata = {"Unnamed: 0", "AAM", input_column}
    assay_columns = [
        column for column in frame.columns if column not in metadata
    ]
    if not assay_columns:
        raise ValueError(f"{system}: no assay columns found")

    for column in assay_columns:
        values = set(frame[column].dropna().tolist())
        if not values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"{system}.{column}: expected binary labels, "
                f"found {sorted(values)}"
            )

    return input_column, input_type, assay_columns


def prepare_datasets(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize source tables into reaction-level and long assay tables."""
    reaction_records: list[dict[str, Any]] = []
    assay_records: list[dict[str, Any]] = []
    system_counts: dict[str, dict[str, int]] = {}

    for system in TABLES:
        frame = tables[system]
        input_column, input_type, assay_columns = _source_schema(frame, system)
        system_assays = 0
        system_active = 0

        for _, source_row in frame.iterrows():
            mapped = str(source_row["AAM"]).strip()
            if (
                not mapped
                or mapped.lower() == "nan"
                or not looks_mapped(mapped)
            ):
                raise ValueError(
                    f"{system}:{source_row['Unnamed: 0']}: "
                    "invalid mapped reaction"
                )
            if ">>" not in mapped:
                raise ValueError(
                    f"{system}:{source_row['Unnamed: 0']}: "
                    "not a reaction SMILES"
                )

            reaction_r_id = len(reaction_records)
            labels = {
                assay: int(source_row[assay])
                for assay in assay_columns
                if pd.notna(source_row[assay])
            }
            unmapped = remove_atom_mapping(mapped)
            if looks_mapped(unmapped):
                raise ValueError(
                    f"{system}:{source_row['Unnamed: 0']}: AAM removal failed"
                )

            shared = {
                "ground_truth": mapped,
                "rxn": unmapped,
                "original_id": source_row["Unnamed: 0"],
                "system": system,
                "input": source_row[input_column],
                "input_type": input_type,
                "mapping_curated": True,
                "validated": None,
            }
            reaction_records.append(
                {
                    "r_id": reaction_r_id,
                    **shared,
                    "assay_labels": json.dumps(labels, sort_keys=True),
                    "n_assays": len(labels),
                    "n_active": sum(labels.values()),
                }
            )

            for assay, activity in labels.items():
                assay_records.append(
                    {
                        "r_id": len(assay_records),
                        "reaction_r_id": reaction_r_id,
                        **shared,
                        "assay": assay,
                        "activity": activity,
                    }
                )
                system_assays += 1
                system_active += activity

        system_counts[system] = {
            "reactions": len(frame),
            "assays": system_assays,
            "active": system_active,
            "inactive": system_assays - system_active,
        }

    reactions = pd.DataFrame(reaction_records, columns=REACTION_COLUMNS)
    assays = pd.DataFrame(assay_records, columns=ASSAY_COLUMNS)
    counts = {
        "reactionCount": len(reactions),
        "uniqueMappedReactionCount": int(reactions["ground_truth"].nunique()),
        "assayCount": len(assays),
        "activeCount": int(assays["activity"].sum()),
        "inactiveCount": int((assays["activity"] == 0).sum()),
        "systems": system_counts,
    }
    return reactions, assays, counts


def save_dataframe(frame: pd.DataFrame, output: Path) -> None:
    """Write a gzipped CSV using the project helper if present."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from synrxn.io.io import save_df_gz

        save_df_gz(frame, output)
    except Exception as exc:
        logging.warning(
            "Project CSV helper unavailable (%s); using pandas",
            exc,
        )
        frame.to_csv(output, index=False, compression="gzip")
    logging.info("Saved %d rows to %s", len(frame), output)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SynRXN datasets from EHreact manuscript assays."
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="EHreact Git ref (default: pinned upstream main commit).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Optional directory containing the nine source CSVs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--assay-output", type=Path, default=DEFAULT_ASSAY_OUTPUT
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="Optional downstream JSON provenance report.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Download timeout in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(
    argv: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    tables, sources = load_tables(
        ref=args.ref,
        input_dir=args.input_dir,
        timeout=args.timeout,
    )
    reactions, assays, counts = prepare_datasets(tables)
    save_dataframe(reactions, args.output)
    save_dataframe(assays, args.assay_output)

    provenance = {
        "builder": Path(__file__).name,
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "ref": args.ref,
        "paper": "https://doi.org/10.1021/acs.jcim.1c00921",
        "license": "BSD-3-Clause",
        "licenseUrl": f"{REPOSITORY}/blob/{args.ref}/LICENSE",
        "copyright": "Copyright 2021 Esther Heid",
        "mappingCuration": (
            "RDT mappings; anomalous reaction centres flagged and mappings "
            "manually corrected according to the EHreact manuscript."
        ),
        "rowLevelValidationField": False,
        "outputs": {
            "reactions": str(args.output),
            "assays": str(args.assay_output),
        },
        "counts": counts,
        "sources": sources,
    }
    if args.provenance_output is not None:
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logging.info("Saved provenance to %s", args.provenance_output)
    logging.info(
        "Built %d reactions and %d reaction-assay labels",
        len(reactions),
        len(assays),
    )
    return reactions, assays


if __name__ == "__main__":
    main()
