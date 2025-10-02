from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterator, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class SplitIndices:
    """
    Container for indices of a single (repeat, fold) split.

    :param repeat: Repeat index (0-based).
    :type repeat: int
    :param fold: Fold index within the repeat (0-based).
    :type fold: int
    :param train_idx: Numpy array of training row indices.
    :type train_idx: numpy.ndarray
    :param val_idx: Numpy array of validation row indices.
    :type val_idx: numpy.ndarray
    :param test_idx: Numpy array of test row indices.
    :type test_idx: numpy.ndarray
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
    The outer holdout fold is further split into validation and test according to the
    supplied ratio (for example, ``ratio=(8,1,1)`` -> holdout split val:test = 1:1).

    .. code-block:: python

        splitter = RepeatedKFoldsSplitter(
            n_splits=5,
            n_repeats=5,
            ratio=(8, 1, 1),
            shuffle=True,
            random_state=42,
        )

        splitter.split(df, stratify_col='label')

        # get numpy index arrays
        train_idx, val_idx, test_idx = splitter.get_split(repeat=0, fold=0, as_frame=False)

        # get DataFrame slices
        train_df, val_df, test_df = splitter.get_split(repeat=0, fold=0, as_frame=True)

        # quick inspection
        print(splitter)           # summary repr
        print(splitter[ (0,0) ])  # SplitIndices for repeat=0, fold=0
        len(splitter)             # number of generated splits

    :param n_splits: Number of outer folds (k).
    :type n_splits: int
    :param n_repeats: Number of repeats (how many times to reshuffle-and-split).
    :type n_repeats: int
    :param ratio: Tuple of three ints (train, val, test) like (8,1,1).
    :type ratio: tuple[int, int, int]
    :param shuffle: Whether to shuffle before splitting each repeat.
    :type shuffle: bool
    :param random_state: Base random state for reproducible repeats.
    :type random_state: Optional[int]
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
        if any(r <= 0 for r in ratio):
            raise ValueError("All entries of ratio must be positive integers")

        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.ratio = (int(ratio[0]), int(ratio[1]), int(ratio[2]))
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self._val_frac_within_holdout = self.ratio[1] / (self.ratio[1] + self.ratio[2])
        self._splits: List[SplitIndices] = []
        self._df: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        return (
            f"RepeatedKFoldsSplitter(n_splits={self.n_splits}, n_repeats={self.n_repeats}, "
            f"ratio={self.ratio}, generated_splits={len(self._splits)}, random_state={self.random_state})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """Return number of generated (repeat, fold) splits."""
        return len(self._splits)

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> SplitIndices:
        """
        Retrieve a SplitIndices either by integer index (0-based over stored splits)
        or by tuple (repeat, fold).

        :param key: int or (repeat, fold)
        :type key: int or tuple[int,int]
        :raises IndexError: If key is out of range or not found.
        :returns: SplitIndices object.
        :rtype: SplitIndices
        """
        if isinstance(key, int):
            return self._splits[key]
        if isinstance(key, tuple) and len(key) == 2:
            repeat, fold = int(key[0]), int(key[1])
            for s in self._splits:
                if s.repeat == repeat and s.fold == fold:
                    return s
            raise IndexError(f"No split for repeat={repeat}, fold={fold}")
        raise TypeError("Key must be int or tuple(repeat, fold)")

    def split(
        self,
        df: pd.DataFrame,
        stratify_col: Optional[Union[str, pd.Series]] = None,
    ) -> List[SplitIndices]:
        """
        Compute and store splits for the provided DataFrame.

        :param df: Input DataFrame (rows will be split).
        :type df: pandas.DataFrame
        :param stratify_col: Column name or Series used to stratify outer folds (optional).
        :type stratify_col: Optional[str or pandas.Series]
        :returns: List of SplitIndices objects for every (repeat, fold).
        :rtype: list[SplitIndices]
        :raises ValueError: If n_splits is larger than dataset size or if stratify length mismatches.
        """
        self._df = df.reset_index(drop=True)
        n = len(self._df)
        if n < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} is larger than dataset size {n}"
            )

        stratify_arr = None
        if stratify_col is not None:
            if isinstance(stratify_col, str):
                stratify_arr = self._df[stratify_col].values
            else:
                stratify_arr = np.asarray(stratify_col)
            if len(stratify_arr) != n:
                raise ValueError(
                    "Length of stratify_col does not match dataframe length"
                )

        self._splits = []
        for r in range(self.n_repeats):
            rs = None if self.random_state is None else int(self.random_state) + r
            if stratify_arr is None:
                kf = KFold(
                    n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs
                )
                split_gen = kf.split(self._df)
            else:
                skf = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs
                )
                split_gen = skf.split(self._df, stratify_arr)

            for fold_i, (_, hold_idx) in enumerate(split_gen):
                mask = np.ones(n, dtype=bool)
                mask[hold_idx] = False
                train_idx = np.nonzero(mask)[0]

                hold_targets = (
                    stratify_arr[hold_idx] if stratify_arr is not None else None
                )
                val_idx, test_idx = train_test_split(
                    hold_idx,
                    test_size=(1.0 - self._val_frac_within_holdout),
                    random_state=rs,
                    shuffle=True,
                    stratify=hold_targets,
                )

                self._splits.append(
                    SplitIndices(
                        repeat=r,
                        fold=fold_i,
                        train_idx=train_idx,
                        val_idx=np.asarray(val_idx, dtype=int),
                        test_idx=np.asarray(test_idx, dtype=int),
                    )
                )

        return self._splits

    def get_split(
        self,
        repeat: int = 0,
        fold: int = 0,
        as_frame: bool = False,
    ) -> Tuple[
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
    ]:
        """
        Retrieve indices or DataFrames for a particular (repeat, fold).

        :param repeat: Repeat index (0-based).
        :type repeat: int
        :param fold: Fold index within the repeat (0-based).
        :type fold: int
        :param as_frame: If True, return (train_df, val_df, test_df) slices; otherwise return index arrays.
        :type as_frame: bool
        :returns: (train, val, test) either as index arrays or DataFrames.
        :rtype: tuple
        :raises RuntimeError: If .split(...) has not been called before.
        :raises IndexError: If requested (repeat, fold) does not exist.
        """
        if not self._splits:
            raise RuntimeError("Call .split(df, ...) before requesting a split")

        for s in self._splits:
            if s.repeat == repeat and s.fold == fold:
                found = s
                break
        else:
            raise IndexError(
                f"No split for repeat={repeat}, fold={fold}. Available repeats: 0..{self.n_repeats-1},"
                + f" folds: 0..{self.n_splits-1}"
            )

        if as_frame:
            assert self._df is not None
            return (
                self._df.iloc[found.train_idx].reset_index(drop=True),
                self._df.iloc[found.val_idx].reset_index(drop=True),
                self._df.iloc[found.test_idx].reset_index(drop=True),
            )
        else:
            return found.train_idx.copy(), found.val_idx.copy(), found.test_idx.copy()

    def iter_splits(self) -> Iterator[SplitIndices]:
        """
        Iterate over computed splits in order (repeat major, fold minor).

        :returns: Iterator of SplitIndices objects.
        :rtype: Iterator[SplitIndices]
        """
        for s in self._splits:
            yield s

    @property
    def splits(self) -> List[SplitIndices]:
        """Return a copy of computed splits."""
        return list(self._splits)

    @property
    def n_generated_splits(self) -> int:
        """Number of generated (repeat, fold) splits."""
        return len(self._splits)
