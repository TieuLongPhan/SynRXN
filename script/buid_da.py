"""
Curation and remapping of TwoBondChem Diels–Alder reactions.

Paper:
    https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00092g

Repository:
    https://github.com/Goodman-lab/TwoBondChem

License:
    The associated article is published under the Creative Commons Attribution
    3.0 Unported License (CC BY 3.0). The GitHub repository does not appear to
    provide a separate explicit license for the dataset files, so redistribution
    of curated derivatives should include proper attribution, a description of
    modifications, and verification of reuse terms when needed.

This module curates the atom-mapped Diels–Alder reactions from the
TwoBondChem dataset. The original reactions may contain incorrect atom
mappings, especially for cases where mappers assigns atoms in a way that
does not preserve the chemically expected Diels–Alder reaction center.

To obtain chemically consistent mappings, each reaction is remapped using two
independent Diels–Alder reaction rules. The remapped candidates are then tested
against the original reaction by applying the inferred transformation and
checking whether the expected product can be reproduced from the reactants.

Only the mapping that successfully reproduces the reaction is retained. If both
rules fail, or if both rules generate ambiguous but different valid mappings,
the reaction is marked for manual inspection or rejected from the curated
dataset.

The curated output is intended for SynRXN reaction-center and ITS
construction, where atom-map consistency is required.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Iterable,
    Literal,
    Optional,
    Sequence,
    overload,
)

import pandas as pd

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO import gml_to_its
from synkit.Synthesis.Reactor.syn_reactor import SynReactor


DATA_URL = (
    "https://raw.githubusercontent.com/Goodman-lab/TwoBondChem/"
    "refs/heads/main/dataset/diels_alder_data_v7_19052024.csv"
)


DEFAULT_OUTPUT = Path("Data/synthesis/da.csv.gz")

REACTION_COLUMN = "reaction"
ORIGINAL_REACTION_COLUMN = "reaction_original"
INFERRED_COLUMN = "rsmi"

GroupedRSMI = dict[str, list[str]]
FailedRSMI = list[tuple[str, Exception]]
RuleLike = Any


RULE_1 = """
rule [
   ruleID "rule"
   left [
      edge [ source 1 target 3 label "=" ]
      edge [ source 2 target 4 label "=" ]
      edge [ source 3 target 5 label "-" ]
      edge [ source 5 target 6 label "=" ]
   ]
   context [
      node [ id 1 label "C" ]
      node [ id 2 label "C" ]
      node [ id 3 label "C" ]
      node [ id 4 label "C" ]
      node [ id 5 label "C" ]
      node [ id 6 label "C" ]
   ]
   right [
      edge [ source 1 target 2 label "-" ]
      edge [ source 1 target 3 label "-" ]
      edge [ source 2 target 4 label "-" ]
      edge [ source 3 target 5 label "=" ]
      edge [ source 4 target 6 label "-" ]
      edge [ source 5 target 6 label "-" ]
   ]
]
"""


RULE_2 = """
rule [
   ruleID "rule"
   left [
      edge [ source 1 target 2 label "=" ]
      edge [ source 2 target 4 label "-" ]
      edge [ source 3 target 5 label "#" ]
      edge [ source 4 target 6 label "=" ]
   ]
   context [
      node [ id 1 label "C" ]
      node [ id 2 label "C" ]
      node [ id 3 label "C" ]
      node [ id 4 label "C" ]
      node [ id 5 label "C" ]
      node [ id 6 label "C" ]
   ]
   right [
      edge [ source 1 target 2 label "-" ]
      edge [ source 1 target 3 label "-" ]
      edge [ source 2 target 4 label "=" ]
      edge [ source 3 target 5 label "=" ]
      edge [ source 4 target 6 label "-" ]
      edge [ source 5 target 6 label "-" ]
   ]
]
"""


@dataclass(frozen=True)
class InferenceStats:
    total_rows: int
    standardized_rows: int
    failed_standardizations: int
    inferred_rows: int
    missing_inference: int


def standardize_rsmi(
    rsmi: str,
    *,
    std: Standardize,
    remove_aam: bool = True,
    ignore_stereo: bool = True,
) -> str:
    """
    Standardize one reaction SMILES.

    Parameters
    ----------
    rsmi : str
        Input reaction SMILES.
    std : Standardize
        Reusable Standardize instance.
    remove_aam : bool, default=True
        Remove atom-atom mapping.
    ignore_stereo : bool, default=True
        Ignore stereochemistry during standardization.

    Returns
    -------
    str
        Standardized reaction SMILES.
    """
    return std.fit(
        rsmi,
        remove_aam=remove_aam,
        ignore_stereo=ignore_stereo,
    )


def group_rsmi(
    results: Iterable[str],
    *,
    std: Optional[Standardize] = None,
    remove_duplicates: bool = True,
    return_failed: bool = False,
    ignore_stereo: bool = True,
) -> GroupedRSMI | tuple[GroupedRSMI, FailedRSMI]:
    """
    Group original atom-mapped reaction SMILES by standardized unmapped RSMI.

    Parameters
    ----------
    results : Iterable[str]
        Atom-mapped reaction SMILES candidates.
    std : Standardize, optional
        Existing Standardize instance. Reusing one is faster.
    remove_duplicates : bool, default=True
        Remove duplicate mapped RSMIs inside each standardized group.
    return_failed : bool, default=False
        If True, also return failed standardizations.
    ignore_stereo : bool, default=True
        Ignore stereochemistry while grouping.

    Returns
    -------
    dict[str, list[str]]
        standardized_unmapped_rsmi -> list of original mapped RSMIs

    tuple[dict[str, list[str]], list[tuple[str, Exception]]]
        Returned only when `return_failed=True`.
    """
    standardizer = std or Standardize()

    grouped: DefaultDict[str, list[str]] = defaultdict(list)
    failed: FailedRSMI = []

    seen: DefaultDict[str, set[str]] | None = (
        defaultdict(set) if remove_duplicates else None
    )

    for mapped_rsmi in results:
        try:
            key = standardize_rsmi(
                mapped_rsmi,
                std=standardizer,
                remove_aam=True,
                ignore_stereo=ignore_stereo,
            )
        except Exception as exc:
            failed.append((mapped_rsmi, exc))
            continue

        if seen is not None:
            if mapped_rsmi in seen[key]:
                continue
            seen[key].add(mapped_rsmi)

        grouped[key].append(mapped_rsmi)

    grouped_dict = dict(grouped)

    if return_failed:
        return grouped_dict, failed

    return grouped_dict


def get_reactants_from_rsmi(rsmi: str) -> str:
    """
    Extract the reactant side from a reaction SMILES.

    Parameters
    ----------
    rsmi : str
        Reaction SMILES.

    Returns
    -------
    str
        Reactant side of the reaction SMILES.

    Raises
    ------
    ValueError
        If `rsmi` is malformed.
    """
    if ">>" not in rsmi:
        raise ValueError(f"Invalid reaction SMILES. Expected '>>' in: {rsmi!r}")

    reactants, _products = rsmi.split(">>", 1)

    if not reactants:
        raise ValueError(f"Invalid reaction SMILES. Empty reactant side: {rsmi!r}")

    return reactants


@overload
def infer(
    rsmi: str,
    rc: RuleLike,
    *,
    std: Optional[Standardize] = None,
    remove_duplicates: bool = True,
    return_all: Literal[True] = True,
    ignore_stereo: bool = True,
) -> list[str] | None: ...


@overload
def infer(
    rsmi: str,
    rc: RuleLike,
    *,
    std: Optional[Standardize] = None,
    remove_duplicates: bool = True,
    return_all: Literal[False],
    ignore_stereo: bool = True,
) -> str | None: ...


def infer(
    rsmi: str,
    rc: RuleLike,
    *,
    std: Optional[Standardize] = None,
    remove_duplicates: bool = True,
    return_all: bool = True,
    ignore_stereo: bool = True,
) -> list[str] | str | None:
    """
    Infer mapped reaction candidates from one reaction center.

    Parameters
    ----------
    rsmi : str
        Target reaction SMILES. Can be mapped or unmapped.
    rc : Any
        Reaction center / rule used by `SynReactor`.
    std : Standardize, optional
        Existing Standardize instance.
    remove_duplicates : bool, default=True
        Remove duplicate mapped candidates.
    return_all : bool, default=True
        If True, return all matching mapped RSMIs.
        If False, return only the first matching mapped RSMI.
    ignore_stereo : bool, default=True
        Ignore stereochemistry during matching.

    Returns
    -------
    list[str] | str | None
        Matching mapped RSMI candidates, one candidate, or None.
    """
    standardizer = std or Standardize()

    target_rsmi = standardize_rsmi(
        rsmi,
        std=standardizer,
        remove_aam=True,
        ignore_stereo=ignore_stereo,
    )

    reactants = get_reactants_from_rsmi(target_rsmi)

    reactor = SynReactor(reactants, rc)

    candidates = getattr(reactor, "smarts", None)
    if candidates is None:
        raise AttributeError("SynReactor object does not expose a `smarts` attribute.")

    grouped = group_rsmi(
        candidates,
        std=standardizer,
        remove_duplicates=remove_duplicates,
        ignore_stereo=ignore_stereo,
    )

    matches = grouped.get(target_rsmi)

    if not matches:
        return None

    return matches if return_all else matches[0]


def infer_with_rules(
    rsmi: str,
    rules: Sequence[RuleLike],
    *,
    std: Standardize,
    ignore_stereo: bool = True,
    remove_duplicates: bool = True,
) -> str | None:
    """
    Try multiple reaction rules in order and return the first inferred candidate.

    Parameters
    ----------
    rsmi : str
        Standardized reaction SMILES.
    rules : Sequence[Any]
        Reaction centers/rules.
    std : Standardize
        Reusable Standardize instance.
    ignore_stereo : bool, default=True
        Ignore stereochemistry during matching.
    remove_duplicates : bool, default=True
        Remove duplicate mapped candidates.

    Returns
    -------
    str | None
        First inferred mapped RSMI, or None.
    """
    for rc in rules:
        candidate = infer(
            rsmi,
            rc,
            std=std,
            remove_duplicates=remove_duplicates,
            return_all=False,
            ignore_stereo=ignore_stereo,
        )

        if candidate is not None:
            return candidate

    return None


def build_rules() -> list[RuleLike]:
    """
    Compile GML rules into ITS reaction-center objects.

    Returns
    -------
    list[Any]
        Compiled reaction-center rules.
    """
    return [
        gml_to_its(RULE_1),
        gml_to_its(RULE_2),
    ]


def load_dataset(path_or_url: str | Path) -> pd.DataFrame:
    """
    Load the Diels-Alder dataset.

    Parameters
    ----------
    path_or_url : str | Path
        Local path or URL to CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    logging.info("Loading dataset from %s", path_or_url)
    return pd.read_csv(path_or_url)


def standardize_dataset_reactions(
    df: pd.DataFrame,
    *,
    reaction_column: str,
    std: Standardize,
    ignore_stereo: bool = True,
    keep_original: bool = True,
) -> tuple[pd.DataFrame, FailedRSMI]:
    """
    Standardize the reaction column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    reaction_column : str
        Name of the reaction column.
    std : Standardize
        Reusable Standardize instance.
    ignore_stereo : bool, default=True
        Ignore stereochemistry during standardization.
    keep_original : bool, default=True
        Preserve the original reaction column as `reaction_original`.

    Returns
    -------
    tuple[pandas.DataFrame, FailedRSMI]
        Updated DataFrame and failed standardization records.
    """
    if reaction_column not in df.columns:
        raise KeyError(
            f"Reaction column {reaction_column!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    if keep_original and ORIGINAL_REACTION_COLUMN not in df.columns:
        df.insert(
            loc=df.columns.get_loc(reaction_column),
            column=ORIGINAL_REACTION_COLUMN,
            value=df[reaction_column],
        )

    standardized: list[str | None] = []
    failed: FailedRSMI = []

    for raw_rsmi in df[reaction_column]:
        if pd.isna(raw_rsmi):
            failed.append(("<NA>", ValueError("Missing reaction SMILES.")))
            standardized.append(None)
            continue

        raw_rsmi_str = str(raw_rsmi)

        try:
            rsmi = standardize_rsmi(
                raw_rsmi_str,
                std=std,
                remove_aam=True,
                ignore_stereo=ignore_stereo,
            )
        except Exception as exc:
            failed.append((raw_rsmi_str, exc))
            standardized.append(None)
            continue

        standardized.append(rsmi)

    df[reaction_column] = standardized

    return df, failed


def infer_dataset_reactions(
    df: pd.DataFrame,
    *,
    reaction_column: str,
    inferred_column: str,
    rules: Sequence[RuleLike],
    std: Standardize,
    ignore_stereo: bool = True,
) -> pd.DataFrame:
    """
    Infer mapped RSMI candidates for each standardized reaction in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    reaction_column : str
        Column containing standardized reaction SMILES.
    inferred_column : str
        Output column for inferred mapped reaction SMILES.
    rules : Sequence[Any]
        Reaction-center rules tried in order.
    std : Standardize
        Reusable Standardize instance.
    ignore_stereo : bool, default=True
        Ignore stereochemistry during inference.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame.
    """
    df = df.copy()
    inferred: list[str | None] = []

    for rsmi in df[reaction_column]:
        if pd.isna(rsmi):
            inferred.append(None)
            continue

        try:
            candidate = infer_with_rules(
                str(rsmi),
                rules,
                std=std,
                ignore_stereo=ignore_stereo,
                remove_duplicates=True,
            )
        except Exception as exc:
            logging.debug("Inference failed for %r: %s", rsmi, exc)
            candidate = None

        inferred.append(candidate)

    df[inferred_column] = inferred

    return df


def prepare_dataset(
    df: pd.DataFrame,
    *,
    reaction_column: str = REACTION_COLUMN,
    inferred_column: str = INFERRED_COLUMN,
    rules: Optional[Sequence[RuleLike]] = None,
    std: Optional[Standardize] = None,
    ignore_stereo: bool = True,
    keep_original: bool = True,
) -> tuple[pd.DataFrame, InferenceStats]:
    """
    Standardize reactions and infer mapped RSMI candidates.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    reaction_column : str, default="reaction"
        Name of the reaction column.
    inferred_column : str, default="rsmi"
        Name of the inferred output column.
    rules : Sequence[Any], optional
        Precompiled reaction-center rules.
    std : Standardize, optional
        Existing Standardize instance.
    ignore_stereo : bool, default=True
        Ignore stereochemistry.
    keep_original : bool, default=True
        Preserve original reaction strings.

    Returns
    -------
    tuple[pandas.DataFrame, InferenceStats]
        Processed DataFrame and summary statistics.
    """
    standardizer = std or Standardize()
    compiled_rules = list(rules) if rules is not None else build_rules()

    total_rows = len(df)

    df, failed_standardizations = standardize_dataset_reactions(
        df,
        reaction_column=reaction_column,
        std=standardizer,
        ignore_stereo=ignore_stereo,
        keep_original=keep_original,
    )

    df = infer_dataset_reactions(
        df,
        reaction_column=reaction_column,
        inferred_column=inferred_column,
        rules=compiled_rules,
        std=standardizer,
        ignore_stereo=ignore_stereo,
    )

    standardized_rows = int(df[reaction_column].notna().sum())
    inferred_rows = int(df[inferred_column].notna().sum())

    stats = InferenceStats(
        total_rows=total_rows,
        standardized_rows=standardized_rows,
        failed_standardizations=len(failed_standardizations),
        inferred_rows=inferred_rows,
        missing_inference=standardized_rows - inferred_rows,
    )

    return df, stats


def save_dataframe(df: pd.DataFrame, output: str | Path) -> None:
    """
    Save a DataFrame as gzipped CSV.

    Uses `synrxn.io.io.save_df_gz` if available. Otherwise falls back
    to `pandas.DataFrame.to_csv(..., compression="gzip")`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    output : str | Path
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
    Optionally import `process_aam` from `process.aam`.

    This is useful when running the script from a project where `process`
    is one level above this file.

    Returns
    -------
    Callable[..., Any] | None
        Imported `process_aam` function, or None if unavailable.
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
    Optionally apply a user-defined `process_aam` function.

    Notes
    -----
    This assumes `process_aam(df)` returns either a DataFrame or None.
    If your local `process_aam` has a different signature, modify this
    wrapper only.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    enabled : bool, default=False
        Whether to apply `process_aam`.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame.
    """
    if not enabled:
        return df

    process_aam = try_import_process_aam()

    if process_aam is None:
        logging.warning("`process_aam` was requested but could not be imported.")
        return df

    logging.info("Applying optional `process_aam` hook.")

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
        description="Infer mapped RSMI candidates for the TwoBondChem DA dataset."
    )

    parser.add_argument(
        "--input",
        default=DATA_URL,
        help="Input CSV path or URL.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output gzipped CSV path.",
    )

    parser.add_argument(
        "--reaction-column",
        default=REACTION_COLUMN,
        help="Column containing reaction SMILES.",
    )

    parser.add_argument(
        "--inferred-column",
        default=INFERRED_COLUMN,
        help="Output column for inferred mapped RSMI.",
    )

    parser.add_argument(
        "--keep-original",
        action="store_true",
        default=True,
        help="Keep original reaction strings in `reaction_original`.",
    )

    parser.add_argument(
        "--drop-original",
        action="store_true",
        help="Do not keep original reaction strings.",
    )

    parser.add_argument(
        "--ignore-stereo",
        action="store_true",
        default=True,
        help="Ignore stereochemistry during standardization and matching.",
    )

    parser.add_argument(
        "--respect-stereo",
        action="store_true",
        help="Respect stereochemistry instead of ignoring it.",
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

    keep_original = args.keep_original and not args.drop_original
    ignore_stereo = args.ignore_stereo and not args.respect_stereo

    standardizer = Standardize()
    rules = build_rules()

    df = load_dataset(args.input)
    df["r_id"] = df["idx"]
    df.drop("idx", axis=1, inplace=True)

    df = optionally_apply_process_aam(
        df,
        enabled=args.process_aam,
    )

    data, stats = prepare_dataset(
        df,
        reaction_column=args.reaction_column,
        inferred_column=args.inferred_column,
        rules=rules,
        std=standardizer,
        ignore_stereo=ignore_stereo,
        keep_original=keep_original,
    )

    logging.info("Total rows: %d", stats.total_rows)
    logging.info("Standardized rows: %d", stats.standardized_rows)
    logging.info("Failed standardizations: %d", stats.failed_standardizations)
    logging.info("Inferred rows: %d", stats.inferred_rows)
    logging.info("Missing inference: %d", stats.missing_inference)

    save_dataframe(data, args.output)

    return data


if __name__ == "__main__":
    main()
