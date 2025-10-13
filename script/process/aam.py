from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from tqdm.auto import tqdm
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.canon_rsmi import CanonRSMI

log = logging.getLogger(__name__)


def process_aam(
    entries: Iterable[Dict[str, Any]],
    *,
    rxn_fn: Optional[Callable[[Any], Any]] = None,
    canon_fn: Optional[Callable[[Any], Any]] = None,
    std: Optional[Standardize] = None,
    canon: Optional[CanonRSMI] = None,
    reactions_key: str = "reactions",
    rxn_key: str = "rxn",
    gt_key: str = "ground_truth",
    inplace: bool = False,
    swallow_exceptions: bool = True,
    return_failures: bool = False,
    progress: bool = True,
    progress_desc: Optional[str] = None,
    progress_disable: Optional[bool] = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    if inplace:
        working: List[Dict[str, Any]] = entries  # type: ignore[assignment]
    else:
        working = [dict(e) for e in entries]

    if rxn_fn is None:
        if std is None:
            std = Standardize()
        rxn_fn = lambda reactions: std.fit(reactions)

    if canon_fn is None:
        if canon is None:
            canon = CanonRSMI(backend="wl", wl_iterations=3)

        def _canon_fn(gt_val: Any) -> Any:
            if gt_val is None:
                return None
            return canon.canonicalise(gt_val).canonical_rsmi

        canon_fn = _canon_fn

    processed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    iterator = working
    if progress:
        total = None
        try:
            total = len(working)  # type: ignore[arg-type]
        except Exception:
            total = None
        iterator = tqdm(
            working,
            desc=(progress_desc or "process_aam"),
            disable=progress_disable,
            total=total,
        )

    for idx, entry in enumerate(iterator):
        try:
            raw_reactions = entry.get(reactions_key)
            entry[rxn_key] = rxn_fn(raw_reactions)
            if gt_key in entry and entry.get(gt_key) is not None:
                entry[gt_key] = canon_fn(entry[gt_key])
        except Exception as exc:
            log.debug(
                "process_aam: error processing entry %s: %s", idx, exc, exc_info=True
            )
            if swallow_exceptions:
                entry[rxn_key] = None
                entry[gt_key] = None
                failures.append({"index": idx, "entry": dict(entry), "error": exc})
            else:
                raise

        if entry.get(rxn_key):
            entry.pop(reactions_key, None)
            processed.append(entry)

    if progress:
        try:
            iterator.close()
        except Exception:
            pass

    if inplace:
        try:
            entries.clear()  # type: ignore[attr-defined]
            entries.extend(processed)  # type: ignore[arg-type]
        except Exception:
            log.debug(
                "Could not replace original 'entries' in-place; returning processed list."
            )

    if return_failures:
        return processed, failures

    return processed
