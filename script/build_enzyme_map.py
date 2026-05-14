"""
Prepare EnzymeMap for SynRXN atom-mapping benchmarks.

Paper:
    https://doi.org/10.1039/D3SC02048G

Data:
    https://zenodo.org/records/8254726

Repository:
    https://github.com/hesther/enzymemap

License:
    The EnzymeMap Zenodo dataset is released under the Creative Commons
    Attribution 4.0 International License (CC BY 4.0). The associated article
    is published under the Creative Commons Attribution 3.0 Unported License
    (CC BY 3.0).

This module downloads the processed EnzymeMap v2 BRENDA 2023 dataset and
converts it into the SynRXN AAM format:

    r_id, ground_truth, rxn, original_id

where ``ground_truth`` is the atom-mapped reaction SMILES and ``rxn`` is the
corresponding unmapped reaction SMILES. The original EnzymeMap row identifier
or raw-reaction identifier is preserved as ``original_id`` when available.

The output is intended for SynRXN atom-mapping, reaction-center, and ITS
construction benchmarks.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd


DATA_URL = (
    "https://zenodo.org/records/8254726/files/"
    "enzymemap_v2_brenda2023.csv.gz?download=1"
)

DEFAULT_OUTPUT = Path("Data/aam/enzyme_map.csv.gz")

OUTPUT_COLUMNS = ["r_id", "ground_truth", "rxn", "original_id"]


MAPPED_COLUMN_CANDIDATES = [
    "mapped",
    "mapped_rxn",
    "mapped_rsmi",
    "mapped_reaction",
    "mapped_reaction_smiles",
    "mapped_rxn_smiles",
    "rxn_mapped",
    "reaction_mapped",
    "atom_mapped_reaction",
    "atom_mapped_reaction_smiles",
]

UNMAPPED_COLUMN_CANDIDATES = [
    "rxn",
    "rsmi",
    "reaction",
    "reaction_smiles",
    "rxn_smiles",
    "unmapped",
    "unmapped_rxn",
    "unmapped_rsmi",
    "unmapped_reaction",
    "unmapped_reaction_smiles",
    "unmapped_rxn_smiles",
]

ORIGINAL_ID_COLUMN_CANDIDATES = [
    "original_id",
    "orig_id",
    "raw_id",
    "raw_idx",
    "raw_index",
    "idx",
    "id",
    "ID",
    "reaction_id",
    "entry_id",
    "brenda_id",
]


def load_dataset(path_or_url: str | Path = DATA_URL) -> pd.DataFrame:
    """
    Load the processed EnzymeMap dataset.

    Parameters
    ----------
    path_or_url : str | pathlib.Path, default=DATA_URL
        Local path or URL to ``enzymemap_v2_brenda2023.csv.gz``.

    Returns
    -------
    pandas.DataFrame
        Loaded EnzymeMap table.
    """
    logging.info("Loading EnzymeMap from %s", path_or_url)
    return pd.read_csv(path_or_url, compression="gzip")


def find_column(
    df: pd.DataFrame, candidates: Sequence[str], *, required: bool
) -> str | None:
    """
    Find a column by trying exact and case-insensitive candidate names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    candidates : Sequence[str]
        Candidate column names.
    required : bool
        If True, raise an error when no candidate is found.

    Returns
    -------
    str | None
        Matched column name, or None if not found and ``required=False``.
    """
    columns = list(df.columns)
    column_lookup = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        matched = column_lookup.get(candidate.lower())
        if matched is not None:
            return matched

    if required:
        raise KeyError(
            "Could not identify required column. "
            f"Tried: {list(candidates)}. "
            f"Available columns: {columns}"
        )

    return None


def looks_mapped(rsmi: str) -> bool:
    """
    Check whether a reaction SMILES appears to contain atom-map numbers.

    Parameters
    ----------
    rsmi : str
        Reaction SMILES.

    Returns
    -------
    bool
        True if atom-map numbers are detected.
    """
    return bool(re.search(r":\d+(?=[^\]]*\])", str(rsmi)))


def remove_atom_mapping_fallback(rsmi: str) -> str:
    """
    Remove atom-map numbers from a reaction SMILES using a regex fallback.

    Notes
    -----
    This fallback preserves the reaction string but does not canonicalize it.
    If ``synkit`` is available, ``remove_atom_mapping`` will use SynKit
    standardization instead.

    Parameters
    ----------
    rsmi : str
        Atom-mapped reaction SMILES.

    Returns
    -------
    str
        Reaction SMILES with atom-map labels removed.
    """
    return re.sub(r":\d+(?=[^\]]*\])", "", str(rsmi))


def remove_atom_mapping(rsmi: str) -> str:
    """
    Remove atom mapping from a reaction SMILES.

    Parameters
    ----------
    rsmi : str
        Atom-mapped reaction SMILES.

    Returns
    -------
    str
        Unmapped reaction SMILES.
    """
    try:
        from synkit.Chem.Reaction.standardize import Standardize

        return Standardize().fit(
            str(rsmi),
            remove_aam=True,
            ignore_stereo=False,
        )
    except Exception:
        return remove_atom_mapping_fallback(str(rsmi))


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert EnzymeMap into SynRXN AAM format.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw EnzymeMap dataframe.

    Returns
    -------
    pandas.DataFrame
        SynRXN-formatted dataframe with columns:

        - ``r_id``
        - ``ground_truth``
        - ``rxn``
        - ``original_id``
    """
    mapped_col = find_column(df, MAPPED_COLUMN_CANDIDATES, required=True)
    unmapped_col = find_column(df, UNMAPPED_COLUMN_CANDIDATES, required=False)
    original_id_col = find_column(df, ORIGINAL_ID_COLUMN_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["r_id"] = range(len(df))
    out["ground_truth"] = df[mapped_col].astype(str)

    if unmapped_col is not None and unmapped_col != mapped_col:
        out["rxn"] = df[unmapped_col].astype(str)
    else:
        logging.info(
            "No separate unmapped reaction column found; removing AAM from mapped RSMI."
        )
        out["rxn"] = out["ground_truth"].map(remove_atom_mapping)

    if original_id_col is not None:
        out["original_id"] = df[original_id_col]
    else:
        out["original_id"] = out["r_id"]

    out = out[OUTPUT_COLUMNS]

    before = len(out)
    out = out.dropna(subset=["ground_truth", "rxn"]).copy()
    out = out[out["ground_truth"].map(looks_mapped)].copy()
    out = out.drop_duplicates(subset=["ground_truth", "rxn"]).reset_index(drop=True)
    out["r_id"] = range(len(out))

    logging.info("Input rows: %d", before)
    logging.info("Output rows after filtering/deduplication: %d", len(out))

    return out


def save_dataframe(df: pd.DataFrame, output: str | Path = DEFAULT_OUTPUT) -> None:
    """
    Save a DataFrame as gzipped CSV.

    Uses ``synrxn.io.io.save_df_gz`` if available, otherwise falls back to
    pandas.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    output : str | pathlib.Path
        Output path.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from synrxn.io.io import save_df_gz

        save_df_gz(df, output)
    except Exception as exc:
        logging.warning(
            "Could not use `synrxn.io.io.save_df_gz`; falling back to pandas. "
            "Reason: %s",
            exc,
        )
        df.to_csv(output, index=False, compression="gzip")

    logging.info("Saved output to %s", output)


def try_import_process_aam() -> Optional[Callable[..., Any]]:
    """
    Optionally import ``process_aam`` from ``process.aam``.

    Returns
    -------
    Callable[..., Any] | None
        Imported ``process_aam`` function, or None if unavailable.
    """
    try:
        from process.aam import process_aam

        return process_aam
    except Exception:
        pass

    try:
        project_root = Path(__file__).resolve().parents[1]
    except Exception:
        return None

    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.append(project_root_str)

    try:
        from process.aam import process_aam

        return process_aam
    except Exception:
        return None


def optionally_apply_process_aam(
    df: pd.DataFrame,
    *,
    enabled: bool = False,
) -> pd.DataFrame:
    """
    Optionally apply local ``process_aam``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    enabled : bool, default=False
        Whether to apply ``process_aam``.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe.
    """
    if not enabled:
        return df

    process_aam = try_import_process_aam()

    if process_aam is None:
        logging.warning("`process_aam` was requested but could not be imported.")
        return df

    result = process_aam(df)

    if result is None:
        return df

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "`process_aam(df)` must return a pandas DataFrame or None. "
            f"Got {type(result)!r}."
        )

    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare EnzymeMap for SynRXN AAM benchmarks."
    )

    parser.add_argument(
        "--input",
        default=DATA_URL,
        help="Input EnzymeMap CSV.GZ path or URL.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output gzipped CSV path.",
    )

    parser.add_argument(
        "--process-aam",
        action="store_true",
        help="Try to import and apply optional `process.aam.process_aam`.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    raw = load_dataset(args.input)
    data = prepare_dataset(raw)

    data = optionally_apply_process_aam(
        data,
        enabled=args.process_aam,
    )

    data = data[OUTPUT_COLUMNS]
    save_dataframe(data, args.output)

    return data


if __name__ == "__main__":
    main()
