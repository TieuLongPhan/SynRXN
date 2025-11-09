from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterator, Union, Iterable, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    train_test_split,
)
from sklearn.utils import check_random_state, indexable


@dataclass
class SplitIndices:
    """
    Container for indices of a single (repeat, fold) split.

    :param repeat: Repeat index (0-based).
    :param fold: Fold index within the repeat (0-based).
    :param train_idx: Numpy array of training row indices.
    :param val_idx: Numpy array of validation row indices.
    :param test_idx: Numpy array of test row indices.
    """

    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def __repr__(self) -> str:
        return (
            f"SplitIndices(repeat={self.repeat}, fold={self.fold}, "
            f"train={len(self.train_idx)}, val={len(self.val_idx)}, test={len(self.test_idx)})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class RepeatedKFoldsSplitter:
    """
    Repeated K-Fold splitter producing (train, val, test) for each outer fold.

    - sklearn-compatible `split(X, y=None, groups=None, stratify=None)` ->
      yields (train_idx, holdout_idx) where holdout_idx == val + test (useful for sklearn `cross_validate`).
    - Use `split_with_val(X, y=None, groups=None, stratify=None)` to receive (train, val, test)
      triples as `SplitIndices` objects (repeat, fold, train_idx, val_idx, test_idx).

    The `ratio` argument is a (train, val, test) tuple that controls the *proportion* used when
    splitting the outer holdout into validation and test sets. For example, `ratio=(8,1,1)`
    means the holdout is split val:test = 1:1.

    Notes on stratification:
    - Pass `y` (array-like) to stratify by labels (sklearn-conventional).
    - Alternatively pass `stratify` to `split(...)` or `split_with_val(...)`. `stratify` may be:
        - a column name (str) when `X` is a pandas.DataFrame, or
        - an array-like of the same length as `X`.
    - If both `y` and `stratify` are provided, `stratify` takes precedence.
    - If stratification is requested but not possible (e.g., a class has fewer than `n_splits` members),
      the splitter falls back to non-stratified `KFold` and emits a warning.

    Example (Sphinx-style)::

        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=500, n_features=20, weights=[0.9, 0.1], random_state=0)
        >>> splitter = RepeatedKFoldsSplitter(n_splits=5, n_repeats=2, ratio=(8,1,1), shuffle=True, random_state=1)
        >>> # sklearn-style outer cross-validation (y is used for stratification)
        >>> for train_idx, hold_idx in splitter.split(X, y):
        ...     print(len(train_idx), len(hold_idx))
        >>> # explicit train/val/test
        >>> for s in splitter.split_with_val(X, stratify=y):
        ...     X_train, X_val, X_test = X[s.train_idx], X[s.val_idx], X[s.test_idx]

    :param n_splits: Number of outer folds (k).
    :param n_repeats: Number of repeats (how many times to reshuffle-and-split).
    :param ratio: Tuple of three ints (train, val, test) like (8,1,1).
    :param shuffle: Whether to shuffle before splitting each repeat.
    :param random_state: Base random state for reproducible repeats.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        ratio: Tuple[int, int, int] = (8, 1, 1),
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if any(int(r) <= 0 for r in ratio):
            raise ValueError("All entries of ratio must be positive integers")
        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.ratio = (int(ratio[0]), int(ratio[1]), int(ratio[2]))
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self._val_frac_within_holdout = self.ratio[1] / (self.ratio[1] + self.ratio[2])
        # computed state
        self._splits: List[SplitIndices] = []
        # store original X if provided as DataFrame for as_frame slicing
        self._X_provided: Optional[Any] = None  # can be DataFrame or array-like

    def __repr__(self) -> str:
        return (
            f"RepeatedKFoldsSplitter(n_splits={self.n_splits}, n_repeats={self.n_repeats}, "
            f"ratio={self.ratio}, generated_splits={len(self._splits)}, random_state={self.random_state})"
        )

    def __len__(self) -> int:
        return len(self._splits)

    # sklearn API
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Return how many (train, holdout) splits will be produced.

        :param X: Feature matrix or dataframe (ignored for counting).
        :param y: Labels (ignored for counting).
        :param groups: Groups (ignored for counting).
        :returns: Total number of outer splits (n_splits * n_repeats).
        """
        return self.n_splits * self.n_repeats

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        stratify: Optional[Union[str, Any]] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        sklearn-compatible generator yielding (train_idx, holdout_idx) where holdout_idx == val + test.

        The `stratify` argument may be:
         - a column name (str) if X is a pandas.DataFrame, or
         - an array-like of length n_samples.

        If `stratify` is provided, it is used in preference to `y`.

        :param X: Feature matrix or pandas.DataFrame.
        :param y: Labels (array-like). Used for stratification if `stratify` is None.
        :param groups: Group labels for GroupKFold (optional).
        :param stratify: Column name or array-like used to stratify folds (optional).
        :yields: Tuples (train_idx, holdout_idx) for each repeat/fold.
        """
        # If a DataFrame column name of stratify is passed, extract it
        stratify_arr = None
        if isinstance(stratify, str):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "When passing stratify as a column name (str), X must be a pandas.DataFrame"
                )
            stratify_arr = X[stratify].values
        elif stratify is not None:
            stratify_arr = np.asarray(stratify)

        # prefer explicit stratify over y
        y_for_split = stratify_arr if stratify_arr is not None else y

        # Ensure X, y_for_split, groups are indexable and consistent lengths
        X_arr, y_arr, groups_arr = indexable(X, y_for_split, groups)
        self._X_provided = X  # store original for get_split(as_frame=True)

        # Ensure splits computed (will use y_arr or groups_arr for stratification/grouping)
        if not self._splits:
            self._compute_splits(X_arr, y_arr, groups_arr)

        for s in self._splits:
            # holdout = val + test (concatenate)
            holdout = np.concatenate([s.val_idx, s.test_idx]).astype(int)
            yield s.train_idx.copy(), holdout.copy()

    def split_with_val(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        stratify: Optional[Union[str, Any]] = None,
    ) -> Iterable[SplitIndices]:
        """
        Yield SplitIndices objects containing (train, val, test) indices.

        :param X: Feature matrix or pandas.DataFrame.
        :param y: Labels (array-like). Used for stratification if `stratify` is None.
        :param groups: Group labels for GroupKFold (optional).
        :param stratify: Column name or array-like used to stratify folds (optional).
        :yields: SplitIndices objects for each repeat and fold.
        """
        # handle stratify similar to split(...)
        stratify_arr = None
        if isinstance(stratify, str):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "When passing stratify as a column name (str), X must be a pandas.DataFrame"
                )
            stratify_arr = X[stratify].values
        elif stratify is not None:
            stratify_arr = np.asarray(stratify)

        y_for_split = stratify_arr if stratify_arr is not None else y

        X_arr, y_arr, groups_arr = indexable(X, y_for_split, groups)
        self._X_provided = X
        if not self._splits:
            self._compute_splits(X_arr, y_arr, groups_arr)
        for s in self._splits:
            yield s

    def _compute_splits(
        self, X_arr: Any, y_arr: Optional[Any], groups_arr: Optional[Any]
    ) -> None:
        """
        Internal: compute and store all SplitIndices for repeats/folds.

        :param X_arr: indexable X (array-like).
        :param y_arr: indexable labels used for stratification (or None).
        :param groups_arr: groups (or None).
        """
        n = len(X_arr)
        if n < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} is larger than dataset size {n}"
            )

        # reset
        self._splits = []
        base_rs = check_random_state(self.random_state)

        # if y_arr is provided use stratification; if groups_arr is provided prefer GroupKFold
        use_groups = groups_arr is not None
        use_stratify = (y_arr is not None) and not use_groups

        for r in range(self.n_repeats):
            rs_seed = base_rs.randint(np.iinfo(np.int32).max)
            rs = check_random_state(int(rs_seed))

            # Choose outer splitter
            if use_groups:
                outer_cv = GroupKFold(n_splits=self.n_splits)
                split_gen = outer_cv.split(X_arr, y_arr, groups_arr)
            elif use_stratify:
                # StratifiedKFold: will raise if a class has fewer than n_splits members
                try:
                    outer_cv = StratifiedKFold(
                        n_splits=self.n_splits,
                        shuffle=self.shuffle,
                        random_state=rs_seed,
                    )
                    split_gen = outer_cv.split(X_arr, y_arr)
                except ValueError:
                    warnings.warn(
                        "StratifiedKFold failed (likely too few members in some classes). Falling back to KFold.",
                        UserWarning,
                    )
                    outer_cv = KFold(
                        n_splits=self.n_splits,
                        shuffle=self.shuffle,
                        random_state=rs_seed,
                    )
                    split_gen = outer_cv.split(X_arr)
            else:
                outer_cv = KFold(
                    n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs_seed
                )
                split_gen = outer_cv.split(X_arr)

            for fold_i, (train_outer_idx, hold_idx) in enumerate(split_gen):
                # train_outer_idx are indices NOT in holdout; however we will recompute 'train' below to ensure
                # train = everything except holdout (consistent with previous API)
                mask = np.ones(n, dtype=bool)
                mask[hold_idx] = False
                train_idx = np.nonzero(mask)[0].astype(int)

                # Now split holdout into val/test according to ratio
                hold_targets = None
                if use_stratify:
                    # For stratification of holdout split, use y_arr[hold_idx] when possible.
                    try:
                        hold_targets = np.asarray(y_arr)[hold_idx]
                    except Exception:
                        hold_targets = None

                try:
                    val_idx, test_idx = train_test_split(
                        hold_idx,
                        test_size=(1.0 - self._val_frac_within_holdout),
                        random_state=int(rs.randint(np.iinfo(np.int32).max)),
                        shuffle=True,
                        stratify=hold_targets,
                    )
                except Exception:
                    warnings.warn(
                        "Inner holdout stratified split failed (likely too few members in some classes). "
                        "Falling back to non-stratified split.",
                        UserWarning,
                    )
                    val_idx, test_idx = train_test_split(
                        hold_idx,
                        test_size=(1.0 - self._val_frac_within_holdout),
                        random_state=int(rs.randint(np.iinfo(np.int32).max)),
                        shuffle=True,
                        stratify=None,
                    )

                self._splits.append(
                    SplitIndices(
                        repeat=r,
                        fold=fold_i,
                        train_idx=np.asarray(train_idx, dtype=int),
                        val_idx=np.asarray(val_idx, dtype=int),
                        test_idx=np.asarray(test_idx, dtype=int),
                    )
                )

    def prepare_splits(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        stratify: Optional[Union[str, Any]] = None,
    ) -> None:
        """
        Compute and store all splits immediately (equivalent to iterating split(...) fully).
        After calling this, self._splits is populated and get_split(...) may be used.
        """
        # handle stratify same as split()
        stratify_arr = None
        if isinstance(stratify, str):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "When passing stratify as a column name (str), X must be a pandas.DataFrame"
                )
            stratify_arr = X[stratify].values
        elif stratify is not None:
            stratify_arr = np.asarray(stratify)

        y_for_split = stratify_arr if stratify_arr is not None else y
        X_arr, y_arr, groups_arr = indexable(X, y_for_split, groups)
        self._X_provided = X
        # compute and save splits
        self._compute_splits(X_arr, y_arr, groups_arr)

    def get_split(self, repeat: int = 0, fold: int = 0, as_frame: bool = False):
        """
        Retrieve either index arrays (train_idx, val_idx, test_idx) or slices
        of the originally provided X (if it was a DataFrame or array-like) when as_frame=True.

        :param repeat: Repeat index (0-based).
        :param fold: Fold index within the repeat (0-based).
        :param as_frame: If True, return slices of the original X (DataFrame or ndarray) rather than indices.
        :returns: Tuple of (train, val, test) either as index arrays or as slices of X.
        :raises RuntimeError: If no splits have been computed yet.
        :raises IndexError: If the requested (repeat, fold) does not exist.
        """
        if not self._splits:
            raise RuntimeError(
                "Call .split(X, ...) or .split_with_val(X, ...) before requesting a split"
            )

        for s in self._splits:
            if s.repeat == repeat and s.fold == fold:
                found = s
                break
        else:
            raise IndexError(f"No split for repeat={repeat}, fold={fold}")

        if as_frame and self._X_provided is not None:
            X = self._X_provided
            # if X is a pandas DataFrame
            if isinstance(X, pd.DataFrame):
                return (
                    X.iloc[found.train_idx].reset_index(drop=True),
                    X.iloc[found.val_idx].reset_index(drop=True),
                    X.iloc[found.test_idx].reset_index(drop=True),
                )
            # if numpy array or list-like -> return slices (np.take is safe)
            try:
                arr = np.asarray(X)
                return (
                    arr[found.train_idx],
                    arr[found.val_idx],
                    arr[found.test_idx],
                )
            except Exception:
                # fallback to returning indices if slicing fails
                return (
                    found.train_idx.copy(),
                    found.val_idx.copy(),
                    found.test_idx.copy(),
                )
        else:
            return found.train_idx.copy(), found.val_idx.copy(), found.test_idx.copy()

    def iter_splits(self) -> Iterator[SplitIndices]:
        """
        Iterate over computed splits in order (repeat major, fold minor).

        :returns: Iterator of SplitIndices objects.
        """
        for s in self._splits:
            yield s

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> SplitIndices:
        """
        Allow indexing into computed splits.

        - splitter[0] -> first stored SplitIndices (by stored-order)
        - splitter[(repeat, fold)] -> SplitIndices for that repeat and fold

        :param key: int or (repeat, fold)
        :raises RuntimeError: if splits have not been computed yet.
        :raises IndexError: if the requested split is not found.
        :raises TypeError: if key type is unsupported.
        :returns: SplitIndices
        """
        if not self._splits:
            raise RuntimeError(
                "No splits computed. Call split(...) or split_with_val(...) before indexing."
            )
        if isinstance(key, int):
            return self._splits[key]
        if isinstance(key, tuple) and len(key) == 2:
            repeat, fold = int(key[0]), int(key[1])
            for s in self._splits:
                if s.repeat == repeat and s.fold == fold:
                    return s
            raise IndexError(f"No split for repeat={repeat}, fold={fold}")
        raise TypeError("Key must be int or tuple(repeat, fold)")

    @property
    def splits(self) -> List[SplitIndices]:
        """Return a copy of computed splits."""
        return list(self._splits)

    @property
    def n_generated_splits(self) -> int:
        """Number of generated (repeat, fold) splits."""
        return len(self._splits)
