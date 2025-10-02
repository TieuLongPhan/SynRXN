from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Iterable
import numpy as np
import pandas as pd
import warnings


class RepeatedGroupAwareSplitter:
    """
    RepeatedGroupAwareSplitter — group/label-aware repeated splitter.

    Treats each unique value in `label_key` as an atomic unit (a group/cluster).
    Rows sharing the same label value will always be kept together.

    :param n_splits: Number of folds (k). Must be >= 2.
    :type n_splits: int
    :param n_repeats: Number of repeated cluster->fold assignments.
    :type n_repeats: int
    :param ratio: Relative proportions (train, val, test). Only val/(val+test) is used when splitting holdout.
    :type ratio: Tuple[int,int,int]
    :param label_key: Column name containing label values. Each unique value defines a group that will not be split.
    :type label_key: str
    :param random_state: Optional integer seed for reproducible assignments.
    :type random_state: Optional[int]
    :param holdout_split_mode: How to split the holdout fold into val/test:
           - "cluster_level" (default): split whole clusters assigned to holdout into val/test,
           - "separate_fold": choose another whole fold as validation.
    :type holdout_split_mode: str
    :param raise_on_leak: If True, fit(...) will raise AssertionError if any split leaks groups.
    :type raise_on_leak: bool
    :param verbose: If True prints diagnostics after fit().
    :type verbose: bool

    Example
    -------
    .. code-block:: python

       import numpy as np
       from repeated_group_aware_splitter import RepeatedGroupAwareSplitter

       data = [
           {"wlfp": np.array([1,0,0]), "group": "A"},
           {"wlfp": np.array([1,0,0]), "group": "A"},
           {"wlfp": np.array([0,1,0]), "group": "B"},
           {"wlfp": np.array([0,0,1]), "group": "C"},
       ]

       splitter = RepeatedGroupAwareSplitter(
           n_splits=3,
           n_repeats=2,
           ratio=(8,1,1),
           label_key="group",
           random_state=0,
           holdout_split_mode="cluster_level",
           raise_on_leak=False,
           verbose=True,
       )

       splitter.fit(data)
       # After fit, results are available as:
       # splitter._all_splits_ok  (bool)
       # splitter._check_results  (list of per-split dicts)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 5,
        ratio: Tuple[int, int, int] = (8, 1, 1),
        label_key: Optional[str] = None,
        random_state: Optional[int] = None,
        holdout_split_mode: str = "cluster_level",
        raise_on_leak: bool = False,
        verbose: bool = False,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if label_key is None:
            raise ValueError(
                "label_key is required (input must contain a label column)"
            )
        if holdout_split_mode not in ("cluster_level", "separate_fold"):
            raise ValueError(
                "holdout_split_mode must be 'cluster_level' or 'separate_fold'"
            )

        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.ratio = tuple(int(x) for x in ratio)
        if sum(self.ratio) <= 0:
            raise ValueError("ratio must sum to a positive value")
        self.label_key = label_key
        self.random_state = None if random_state is None else int(random_state)
        self.holdout_split_mode = holdout_split_mode
        self.raise_on_leak = bool(raise_on_leak)
        self.verbose = bool(verbose)

        # internal
        self._orig_index: Optional[pd.Index] = None
        self._orig_df: Optional[pd.DataFrame] = (
            None  # stored original DataFrame (copy) for label lookups
        )
        self._clusters: Optional[List[List[int]]] = (
            None  # list of clusters, clusters = lists of integer positions (0..n-1)
        )
        self._assignments_per_repeat: Optional[List[Dict[int, int]]] = None
        self._label_values: Optional[List[Any]] = None
        self._label_to_positions: Optional[Dict[Any, List[int]]] = None

        # results from auto-check run at fit-time
        self._check_results: Optional[List[Dict[str, Any]]] = None
        self._all_splits_ok: Optional[bool] = None

    # ---------------- public ----------------
    def fit(self, data: Iterable[Dict] | pd.DataFrame) -> "RepeatedGroupAwareSplitter":
        """
        Fit the splitter on `data`.

        :param data: pandas.DataFrame or iterable of dict-like records. Must contain `label_key`.
        :type data: Union[pd.DataFrame, Iterable[dict]]
        :returns: self
        """
        # normalize input to DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(list(data))

        if self.label_key not in df.columns:
            raise KeyError(f"label_key '{self.label_key}' not found in input data")

        # preserve original DataFrame (copy) for label lookups and to return original indices
        self._orig_df = df.copy()
        self._orig_index = df.index.copy()

        # use reset copy for integer positions mapping 0..n-1
        df_reset = df.reset_index(drop=True)

        # group by label_key => each unique value is an atomic unit
        label_indices = df_reset.groupby(self.label_key).indices
        label_values = list(label_indices.keys())
        if len(label_values) == 0:
            raise ValueError("No labels found in the provided data")

        # map each label to positions (0..n-1)
        label_to_positions: Dict[Any, List[int]] = {}
        for lbl in label_values:
            label_to_positions[lbl] = list(map(int, label_indices[lbl].tolist()))

        # each label becomes one cluster (the cluster is the union of all row positions with that label)
        clusters: List[List[int]] = [
            sorted(label_to_positions[lbl]) for lbl in label_values
        ]

        self._label_values = label_values
        self._label_to_positions = label_to_positions
        self._clusters = clusters

        if len(self._clusters) < self.n_splits:
            warnings.warn(
                f"Number of clusters ({len(self._clusters)}) < n_splits ({self.n_splits}). Some folds may be empty.",
                UserWarning,
            )

        # repeated deterministic assignments cluster_id -> fold_id
        self._assignments_per_repeat = []
        rng_base = np.random.RandomState(self.random_state)
        for rep in range(self.n_repeats):
            seed = rng_base.randint(0, 2**31 - 1)
            rng = np.random.RandomState(seed)
            assignment = self._assign_clusters_to_folds(self._clusters, rng)
            self._assignments_per_repeat.append(assignment)

        # auto-check for group leakage across repeats/folds
        overall_ok, results = self.check_group_separation(raise_on_failure=False)
        self._check_results = results
        self._all_splits_ok = overall_ok

        if not overall_ok:
            # prepare short summary
            n_fails = sum(0 if r["ok"] else 1 for r in results)
            first_fail = next((r for r in results if not r["ok"]), None)
            if self.raise_on_leak:
                raise AssertionError(
                    f"Group leakage detected in {n_fails} splits (first at rep={first_fail['repeat']}"
                    + f" fold={first_fail['fold']}): "
                    f"train&val={first_fail['train&val']}, train&test={first_fail['train&test']},"
                    + f" val&test={first_fail['val&test']}"
                )
            else:
                warnings.warn(
                    f"Group leakage detected in {n_fails} splits (first at rep={first_fail['repeat']}"
                    + f" fold={first_fail['fold']}): "
                    f"train&val={first_fail['train&val']}, train&test={first_fail['train&test']}, "
                    + f"val&test={first_fail['val&test']}",
                    UserWarning,
                )

        if self.verbose:
            self.print_summary()
        return self

    def print_summary(self) -> None:
        """Print concise diagnostics (cluster counts and fold counts for first repeat)."""
        if (
            self._clusters is None
            or self._assignments_per_repeat is None
            or self._orig_index is None
        ):
            print("No fitted state yet.")
            return
        print("RepeatedGroupAwareSplitter summary")
        print("  n_samples:", len(self._orig_index))
        print(
            "  n_labels:",
            len(self._label_values) if self._label_values is not None else 0,
        )
        print("  n_clusters:", len(self._clusters))
        sizes = sorted([len(c) for c in self._clusters], reverse=True)
        print("  cluster sizes (desc, top 10):", sizes[:10])
        assign0 = self._assignments_per_repeat[0]
        fold_counts = [0] * self.n_splits
        for cid, clu in enumerate(self._clusters):
            f = assign0[cid]
            fold_counts[f] += len(clu)
        print("  fold_counts (first repeat):", fold_counts)
        print("  all_splits_ok:", bool(self._all_splits_ok))

    def get_split(
        self, repeat: int, holdout_fold: int
    ) -> Tuple[pd.Index, pd.Index, pd.Index]:
        """
        Return (train_idx, val_idx, test_idx) as pandas.Index into the original input.

        :param repeat: repeat index (0..n_repeats-1)
        :param holdout_fold: which fold is held out and split into val/test (0..n_splits-1)
        :returns: (train_index, val_index, test_index)
        """
        if (
            self._clusters is None
            or self._assignments_per_repeat is None
            or self._orig_index is None
        ):
            raise RuntimeError("Call fit(...) before get_split(...)")
        if not (0 <= repeat < self.n_repeats):
            raise IndexError("repeat out of range")
        if not (0 <= holdout_fold < self.n_splits):
            raise IndexError("holdout_fold out of range")

        assignment = self._assignments_per_repeat[repeat]

        # map integer row position -> fold id
        pos_to_fold: Dict[int, int] = {}
        for cid, cluster in enumerate(self._clusters):
            f = assignment[cid]
            for pos in cluster:
                pos_to_fold[pos] = f

        train_pos = [p for p, f in pos_to_fold.items() if f != holdout_fold]
        # holdout_pos = [p for p, f in pos_to_fold.items() if f == holdout_fold]

        # split holdout into val/test
        if self.holdout_split_mode == "cluster_level":
            val_pos, test_pos = self._split_holdout_clusters(
                repeat, assignment, holdout_fold
            )
        else:  # separate_fold
            val_pos, test_pos = self._separate_fold_val_test(
                repeat, assignment, holdout_fold
            )

        # map integer positions back to original index labels
        train_idx = self._orig_index.take(sorted(train_pos))
        val_idx = self._orig_index.take(sorted(val_pos))
        test_idx = self._orig_index.take(sorted(test_pos))
        return train_idx, val_idx, test_idx

    def get_split_arrays(
        self, repeat: int, holdout_fold: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return numpy arrays of integer positions (0..n-1) for train/val/test."""
        train_idx, val_idx, test_idx = self.get_split(repeat, holdout_fold)
        pos_train = np.asarray(
            [self._orig_index.get_loc(lbl) for lbl in train_idx], dtype=int
        )
        pos_val = np.asarray(
            [self._orig_index.get_loc(lbl) for lbl in val_idx], dtype=int
        )
        pos_test = np.asarray(
            [self._orig_index.get_loc(lbl) for lbl in test_idx], dtype=int
        )
        return pos_train, pos_val, pos_test

    def split_generator(self) -> Iterable[Tuple[pd.Index, pd.Index, pd.Index]]:
        """Yield (train_idx, val_idx, test_idx) for every repeat and fold in order."""
        if self._assignments_per_repeat is None:
            raise RuntimeError("Call fit(...) before split_generator()")
        for rep in range(self.n_repeats):
            for fold in range(self.n_splits):
                yield self.get_split(rep, fold)

    # ---------------- New: integrated check ----------------
    def check_group_separation(
        self, raise_on_failure: bool = False
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check that groups (unique label_key values) are disjoint across train/val/test
        for every (repeat, fold).

        :param raise_on_failure: If True raise AssertionError on first failing split.
        :returns: (ok, results) where `ok` is True if all splits passed, and `results`
                  is a list of dicts with keys:
                    - 'repeat', 'fold', 'ok' (bool),
                    - 'train_labels','val_labels','test_labels' (sets),
                    - 'train&val','train&test','val&test' (intersection sets)
        """
        if self._orig_df is None:
            raise RuntimeError("Call fit(...) before check_group_separation(...)")
        df = self._orig_df

        failures: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []

        for rep in range(self.n_repeats):
            for fold in range(self.n_splits):
                tr_idx, va_idx, te_idx = self.get_split(rep, fold)

                # use .loc with the returned pandas.Index to extract label sets
                tr_labels = set(df.loc[tr_idx, self.label_key].tolist())
                va_labels = set(df.loc[va_idx, self.label_key].tolist())
                te_labels = set(df.loc[te_idx, self.label_key].tolist())

                inter_tr_va = tr_labels & va_labels
                inter_tr_te = tr_labels & te_labels
                inter_va_te = va_labels & te_labels

                ok = not (bool(inter_tr_va) or bool(inter_tr_te) or bool(inter_va_te))

                rec: Dict[str, Any] = {
                    "repeat": rep,
                    "fold": fold,
                    "ok": ok,
                    "train_labels": tr_labels,
                    "val_labels": va_labels,
                    "test_labels": te_labels,
                    "train&val": inter_tr_va,
                    "train&test": inter_tr_te,
                    "val&test": inter_va_te,
                }
                results.append(rec)
                if not ok:
                    failures.append(rec)
                    if raise_on_failure:
                        raise AssertionError(
                            f"Leak detected at rep={rep} fold={fold}: "
                            f"train&val={inter_tr_va}, train&test={inter_tr_te}, val&test={inter_va_te}"
                        )

        overall_ok = len(failures) == 0
        return overall_ok, results

    # ---------------- internal helpers ----------------
    def _assign_clusters_to_folds(
        self, clusters: List[List[int]], rng: np.random.RandomState
    ) -> Dict[int, int]:
        """
        Seed k folds with the k largest clusters (if available), then greedily assign remaining
        clusters to the currently smallest fold.
        """
        k = self.n_splits
        fold_counts = [0] * k
        assignment: Dict[int, int] = {}

        cluster_ids = list(range(len(clusters)))
        cluster_ids.sort(key=lambda cid: -len(clusters[cid]))

        n_clusters = len(cluster_ids)
        m = min(k, n_clusters)
        seed_clusters = cluster_ids[:m]
        rng.shuffle(seed_clusters)
        for fold_idx, cid in enumerate(seed_clusters):
            assignment[cid] = int(fold_idx)
            fold_counts[fold_idx] += len(clusters[cid])

        remaining = [cid for cid in cluster_ids if cid not in seed_clusters]
        for cid in remaining:
            min_count = min(fold_counts)
            candidate_folds = [f for f, c in enumerate(fold_counts) if c == min_count]
            chosen = int(rng.choice(candidate_folds))
            assignment[cid] = chosen
            fold_counts[chosen] += len(clusters[cid])

        return assignment

    def _split_holdout_clusters(
        self, repeat: int, assignment: Dict[int, int], holdout_fold: int
    ) -> Tuple[List[int], List[int]]:
        """Partition clusters assigned to holdout_fold into val/test by whole-cluster granularity."""
        holdout_cluster_ids = [
            cid for cid, f in assignment.items() if f == holdout_fold
        ]
        if not holdout_cluster_ids:
            return [], []
        total_items = sum(len(self._clusters[cid]) for cid in holdout_cluster_ids)
        val_part, test_part = self.ratio[1], self.ratio[2]
        if (val_part + test_part) == 0:
            val_clusters = holdout_cluster_ids[:]
            test_clusters: List[int] = []
        else:
            target_val = total_items * (val_part / (val_part + test_part))
            sorted_cids = sorted(
                holdout_cluster_ids, key=lambda cid: -len(self._clusters[cid])
            )
            seed = (self.random_state or 0) + repeat * 1009 + holdout_fold * 13
            rng = np.random.RandomState(seed)
            # deterministic shuffle within equal-size groups
            i = 0
            while i < len(sorted_cids):
                size = len(self._clusters[sorted_cids[i]])
                j = i + 1
                while (
                    j < len(sorted_cids) and len(self._clusters[sorted_cids[j]]) == size
                ):
                    j += 1
                slice_ids = sorted_cids[i:j]
                rng.shuffle(slice_ids)
                sorted_cids[i:j] = slice_ids
                i = j
            val_clusters: List[int] = []
            test_clusters: List[int] = []
            val_count = 0
            for cid in sorted_cids:
                if abs((val_count + len(self._clusters[cid])) - target_val) < abs(
                    val_count - target_val
                ):
                    val_clusters.append(cid)
                    val_count += len(self._clusters[cid])
                else:
                    test_clusters.append(cid)
        val_pos = [p for cid in val_clusters for p in self._clusters[cid]]
        test_pos = [p for cid in test_clusters for p in self._clusters[cid]]
        return val_pos, test_pos

    def _separate_fold_val_test(
        self, repeat: int, assignment: Dict[int, int], holdout_fold: int
    ) -> Tuple[List[int], List[int]]:
        """Pick another entire fold as validation (no overlap with holdout/test)."""
        pos_to_fold = {}
        for cid, cluster in enumerate(self._clusters):
            f = assignment[cid]
            for pos in cluster:
                pos_to_fold[pos] = f
        candidates = [f for f in range(self.n_splits) if f != holdout_fold]
        if not candidates:
            return [], []
        fold_counts = {f: 0 for f in candidates}
        for pos, f in pos_to_fold.items():
            if f in fold_counts:
                fold_counts[f] += 1
        min_count = min(fold_counts.values())
        min_folds = [f for f, c in fold_counts.items() if c == min_count]
        val_fold = min_folds[repeat % len(min_folds)]
        val_pos = [pos for pos, f in pos_to_fold.items() if f == val_fold]
        test_pos = [pos for pos, f in pos_to_fold.items() if f == holdout_fold]
        return val_pos, test_pos
