from typing import Optional, List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def clean_synrbl(
    data: pd.DataFrame,
    std: Optional[Any] = None,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean a SynRBL dataframe and return a standardized DataFrame.

    :param data: Input dataframe. Must contain an 'expected_reaction' column.
                 May optionally contain 'reaction' and 'id' columns.
    :type data: pd.DataFrame
    :param std: An instance exposing a ``.fit(x)`` method used to standardize reactions.
                If ``None``, a new ``synkit.Chem.Reaction.standardize.Standardize``
                instance will be created internally.
    :type std: object or None
    :param drop_cols: Columns to drop if present before processing. Defaults to
                      ``['R-ids', 'wrong_reactions']`` when ``None``.
    :type drop_cols: list[str] or None

    :returns: A new DataFrame with one row per input row (after dropping rows with
              missing ``expected_reaction``). The columns are:
                - ``R-id`` : value from input ``id`` column if present, otherwise the input index
                - ``rxn`` : result of ``std.fit(reaction)`` or ``None`` on failure / missing reaction
                - ``ground_truth`` : result of ``std.fit(expected_reaction)`` or ``None`` on failure
                - ``error`` : ``dict`` or ``None``. If errors occurred, it contains keys
                  like ``'ground_truth'`` and/or ``'reaction'`` with the error message.
    :rtype: pd.DataFrame

    :raises ValueError: if the required column ``expected_reaction`` is missing.
    """

    if std is None:
        from synkit.Chem.Reaction.standardize import Standardize

        std = Standardize()

    if drop_cols is None:
        drop_cols = ["R-ids", "wrong_reactions"]

    df = data.copy()

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if "expected_reaction" not in df.columns:
        raise ValueError("Input dataframe must contain an 'expected_reaction' column.")
    df = df.dropna(subset=["expected_reaction"])

    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        rec: Dict[str, Any] = {}

        rec["R-id"] = row.get("id", idx)

        gt_raw = row["expected_reaction"]
        gt_err: Optional[str] = None
        try:
            rec["ground_truth"] = std.fit(gt_raw)
        except Exception as exc:
            logger.exception(
                "Standardization failed for ground_truth at R-id=%s", rec["R-id"]
            )
            rec["ground_truth"] = None
            gt_err = str(exc)

        # reaction standardization (may be missing)
        rxn_raw = row.get("reaction") if "reaction" in df.columns else None
        rxn_err: Optional[str] = None
        if pd.isna(rxn_raw) or rxn_raw is None:
            rec["rxn"] = None
        else:
            try:
                rec["rxn"] = std.fit(rxn_raw)
            except Exception as exc:
                logger.exception(
                    "Standardization failed for reaction at R-id=%s", rec["R-id"]
                )
                rec["rxn"] = None
                rxn_err = str(exc)

        # attach error dict only when there were errors
        errors: Dict[str, str] = {}
        if gt_err:
            errors["ground_truth"] = gt_err
        if rxn_err:
            errors["reaction"] = rxn_err
        rec["error"] = errors if errors else None

        records.append(rec)

    result_df = pd.DataFrame.from_records(records)

    # ensure deterministic column order and presence
    cols_order = ["R-id", "rxn", "ground_truth", "error"]
    for c in cols_order:
        if c not in result_df.columns:
            result_df[c] = None
    result_df = result_df[cols_order]

    return result_df
