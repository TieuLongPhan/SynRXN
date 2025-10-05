from typing import Sequence, Union, Any, Set, Optional
from collections.abc import Sequence as _Seq
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_candidates(x: Any, std_inst: Optional[object] = None) -> Set[str]:
    """Convert an element into a set of (optionally standardized) string candidates."""
    if x is None:
        return set()
    if isinstance(x, str):
        vals = (x,)
    elif isinstance(x, (list, tuple, set)):
        vals = tuple(x)
    else:
        # fallback: convert single non-sequence item to str
        return {str(x)}

    out = set()
    for el in vals:
        if el is None:
            continue
        s = str(el)
        if std_inst is not None:
            try:
                s = std_inst.fit(s)
            except Exception as exc:
                # log and fall back to original string
                logger.exception("Standardization failed for candidate %r: %s", el, exc)
                s = str(el)
        out.add(s)
    return out


def acc_rbl(
    ground_truth: Sequence[Union[str, Sequence[str]]],
    pred: Sequence[Union[str, Sequence[str]]],
    std: bool = False,
) -> float:
    """
    Compute exact-match accuracy for RBL-style ground truths and predictions.

    Each item in `ground_truth` and `pred` may be:
      - a string, or
      - a sequence of strings (interpreted as alternative acceptable/predicted values).

    A position is counted correct when any ground-truth candidate exactly equals any predicted candidate.

    :param ground_truth: sequence of ground-truth items (str or sequence of str)
    :param pred: sequence of prediction items (str or sequence of str)
    :param std: if True, standardize every candidate via
                synkit.Chem.Reaction.standardize.Standardize().fit(...) before comparison.
    :returns: accuracy (float between 0.0 and 1.0)
    :raises ValueError: if `ground_truth` and `pred` lengths differ.
    """
    if len(ground_truth) != len(pred):
        raise ValueError("`ground_truth` and `pred` must have the same length.")

    std_inst = None
    if std:
        try:
            from synkit.Chem.Reaction.standardize import Standardize

            std_inst = Standardize()
        except Exception as exc:
            logger.exception(
                "Failed to create Standardize instance; continuing without std: %s", exc
            )
            std_inst = None

    n = len(ground_truth)
    if n == 0:
        return 0.0

    correct = 0
    for gt_item, pred_item in zip(ground_truth, pred):
        gt_cands = _to_candidates(gt_item, std_inst)
        pred_cands = _to_candidates(pred_item, std_inst)
        if gt_cands & pred_cands:
            correct += 1

    return correct / n


def _is_sequence_of_items(x: Any) -> bool:
    """
    True for sequence-like containers (list/tuple/set/np.ndarray/pd.Series/pd.Index)
    but NOT for str/bytes.
    """
    return isinstance(x, _Seq) and not isinstance(x, (str, bytes))


def _first_nonempty_from_sequence(x: Any) -> Optional[str]:
    """
    Iterate sequence-like x and return the first non-empty stripped string,
    or None if nothing found.
    """
    if x is None:
        return None

    # handle pandas Series / numpy arrays / lists / tuples / sets / Index
    if _is_sequence_of_items(x):
        for el in x:
            if el is None:
                continue
            s = str(el).strip()
            if s:
                return s
        return None

    # fallback for scalars
    s = str(x).strip()
    return s if s else None


def _normalize_cell_to_string(x: Any, choose_first: bool = True) -> Optional[str]:
    """
    Normalize a single cell value into a single string (or None).
    - If x is None -> None
    - If x is str -> stripped string or None if empty
    - If x is sequence-like -> return first non-empty candidate (if choose_first True)
      otherwise stringify the whole sequence (not recommended)
    - If x is other scalar -> str(x).strip() or None
    """
    if x is None:
        return None

    if isinstance(x, str):
        s = x.strip()
        return s if s else None

    if _is_sequence_of_items(x):
        if choose_first:
            return _first_nonempty_from_sequence(x)
        # when not choosing first, coerce to string (join-like fallback)
        try:
            # attempt to join by ' | ' for clarity
            seq = [str(el).strip() for el in x if el is not None]
            seq = [s for s in seq if s]
            return " | ".join(seq) if seq else None
        except Exception:
            return None

    # scalars (int/float/other)
    s = str(x).strip()
    return s if s else None


def acc_aam(
    ground_truth: Union[Sequence[Union[str, Sequence[str]]], str],
    pred: Union[Sequence[Union[str, Sequence[str]]], str],
    mapper_name: str = "rxn_mapper",
    validator: Optional[Any] = None,
    choose_first: bool = True,
) -> float:
    """
    Compute AAM-based accuracy using synkit.Chem.Reaction.aam_validator.AAMValidator.

    Top-level inputs may be:
      - a pandas Series / numpy array / list of items (batch),
      - or a single item (string or sequence of alternatives).

    Each element (item) can be:
      - a string (single candidate), or
      - a sequence of candidate strings (we pick the first non-empty one when choose_first=True).

    :param ground_truth: sequence-of-items (or single item) for ground truth
    :param pred: sequence-of-items (or single item) for predictions
    :param mapper_name: column name for predictions in the DataFrame passed to the validator
    :param validator: optional pre-created AAMValidator instance to reuse
    :param choose_first: whether to pick the first non-empty candidate from sequence-like items
    :returns: accuracy as returned by AAMValidator (float between 0.0 and 1.0)
    :raises ValueError: if top-level lengths differ (when both are sequences after normalization)
    :raises ImportError: if AAMValidator cannot be imported/instantiated and no validator provided
    """

    # 1) Convert common column-like top-level inputs to plain Python lists
    if isinstance(ground_truth, (pd.Series, pd.Index)):
        ground_truth = ground_truth.tolist()
    elif isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    if isinstance(pred, (pd.Series, pd.Index)):
        pred = pred.tolist()
    elif isinstance(pred, np.ndarray):
        pred = pred.tolist()

    # 2) Normalize top-level: treat strings as single items (not sequences of chars)
    if _is_sequence_of_items(ground_truth):
        gt_seq = list(ground_truth)
    else:
        gt_seq = [ground_truth]

    if _is_sequence_of_items(pred):
        pred_seq = list(pred)
    else:
        pred_seq = [pred]

    if len(gt_seq) != len(pred_seq):
        raise ValueError(
            "`ground_truth` and `pred` must have the same top-level length."
        )

    # 3) Build lists of single-string cells expected by AAMValidator
    gt_list = []
    pred_list = []
    for gt_item, pred_item in zip(gt_seq, pred_seq):
        gt_val = _normalize_cell_to_string(gt_item, choose_first=choose_first)
        pred_val = _normalize_cell_to_string(pred_item, choose_first=choose_first)
        gt_list.append(gt_val)
        pred_list.append(pred_val)

    df = pd.DataFrame({"ground_truth": gt_list, mapper_name: pred_list})

    # 4) instantiate validator if not provided
    if validator is None:
        try:
            from synkit.Chem.Reaction.aam_validator import AAMValidator

            validator = AAMValidator()
        except Exception as exc:
            logger.exception("Failed to import or instantiate AAMValidator: %s", exc)
            raise ImportError(
                "Could not create AAMValidator; ensure synkit is installed and importable."
            ) from exc

    # 5) Call validator and extract accuracy
    try:
        results = validator.validate_smiles(df, "ground_truth", [mapper_name])
        if not results or not isinstance(results, (list, tuple)):
            raise RuntimeError(
                "Unexpected return value from validator.validate_smiles()"
            )
        accuracy = results[0].get("accuracy")
        if accuracy is None:
            raise KeyError("No 'accuracy' key found in validator result[0].")
        return float(accuracy)
    except Exception as exc:
        logger.exception("AAM validation failed: %s", exc)
        raise
