import unittest
import numpy as np
import pandas as pd

from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter, SplitIndices


class TestRepeatedKFoldsSplitter(unittest.TestCase):
    def test_init_invalid_args(self):
        # n_splits < 2 -> ValueError
        with self.assertRaises(ValueError):
            RepeatedKFoldsSplitter(n_splits=1)

        # ratio entries must be positive
        with self.assertRaises(ValueError):
            RepeatedKFoldsSplitter(n_splits=3, ratio=(8, 0, 1))

    def test_split_error_when_n_splits_larger_than_dataset(self):
        df = pd.DataFrame({"a": list(range(3))})
        splitter = RepeatedKFoldsSplitter(n_splits=5, n_repeats=1)
        with self.assertRaises(ValueError):
            splitter.split(df)

    def test_basic_split_and_get_split(self):
        n = 10
        df = pd.DataFrame({"x": np.arange(n)})
        splitter = RepeatedKFoldsSplitter(
            n_splits=5, n_repeats=1, ratio=(8, 1, 1), shuffle=True, random_state=42
        )
        splits = splitter.split(df)  # returns list[SplitIndices]

        # number of generated splits equals n_splits * n_repeats
        self.assertEqual(len(splits), splitter.n_splits * splitter.n_repeats)
        self.assertEqual(len(splits), splitter.n_generated_splits)
        self.assertEqual(len(splitter), len(splits))

        # check each split: train/val/test disjoint and union covers all indices
        full_set = set(range(n))
        for s in splits:
            self.assertIsInstance(s, SplitIndices)
            train_set = set(s.train_idx.tolist())
            val_set = set(s.val_idx.tolist())
            test_set = set(s.test_idx.tolist())

            # disjoint
            self.assertTrue(train_set.isdisjoint(val_set))
            self.assertTrue(train_set.isdisjoint(test_set))
            self.assertTrue(val_set.isdisjoint(test_set))

            # union equals full set
            union = train_set.union(val_set).union(test_set)
            self.assertEqual(union, full_set)

            # sizes: val + test = holdout size (n / n_splits)
            holdout_size = len(val_set) + len(test_set)
            # for n=10, n_splits=5 -> holdout_size is 2 each fold
            self.assertEqual(holdout_size, n // splitter.n_splits)

        # Test get_split returning index arrays
        train_idx, val_idx, test_idx = splitter.get_split(
            repeat=0, fold=0, as_frame=False
        )
        self.assertIsInstance(train_idx, np.ndarray)
        self.assertIsInstance(val_idx, np.ndarray)
        self.assertIsInstance(test_idx, np.ndarray)

        # Test get_split as_frame returns DataFrames and they match the indices
        train_df, val_df, test_df = splitter.get_split(repeat=0, fold=0, as_frame=True)
        # Reset index on original slices to compare content
        orig_train = df.iloc[train_idx].reset_index(drop=True)
        self.assertTrue(orig_train.equals(train_df))
        orig_val = df.iloc[val_idx].reset_index(drop=True)
        self.assertTrue(orig_val.equals(val_df))
        orig_test = df.iloc[test_idx].reset_index(drop=True)
        self.assertTrue(orig_test.equals(test_df))

    def test_getitem_and_indexing_behaviour(self):
        df = pd.DataFrame({"x": np.arange(12)})
        splitter = RepeatedKFoldsSplitter(
            n_splits=3, n_repeats=2, shuffle=True, random_state=7
        )
        splitter.split(df)

        # integer indexing returns SplitIndices
        first = splitter[0]
        self.assertIsInstance(first, SplitIndices)

        # tuple (repeat, fold)
        s_0_0 = splitter[(0, 0)]
        self.assertIsInstance(s_0_0, SplitIndices)
        self.assertEqual(s_0_0.repeat, 0)
        self.assertEqual(s_0_0.fold, 0)

        # invalid tuple -> IndexError
        with self.assertRaises(IndexError):
            splitter[(999, 999)]

        # invalid key type -> TypeError
        with self.assertRaises(TypeError):
            _ = splitter["bad_key"]

    def test_iter_splits_and_splits_property(self):
        df = pd.DataFrame({"x": np.arange(6)})
        splitter = RepeatedKFoldsSplitter(n_splits=3, n_repeats=1, random_state=1)
        splitter.split(df)

        iter_list = list(splitter.iter_splits())
        prop_list = splitter.splits
        self.assertEqual(len(iter_list), len(prop_list))
        for a, b in zip(iter_list, prop_list):
            self.assertEqual(a.repeat, b.repeat)
            self.assertEqual(a.fold, b.fold)
            np.testing.assert_array_equal(a.train_idx, b.train_idx)
            np.testing.assert_array_equal(a.val_idx, b.val_idx)
            np.testing.assert_array_equal(a.test_idx, b.test_idx)

    def test_stratify_length_mismatch_raises(self):
        df = pd.DataFrame({"x": np.arange(10)})
        splitter = RepeatedKFoldsSplitter(n_splits=5, n_repeats=1)
        # stratify series of wrong length
        with self.assertRaises(ValueError):
            splitter.split(df, stratify_col=pd.Series(np.arange(5)))

    def test_stratified_splits_preserve_label_proportions_in_holdout(self):
        # Use larger class counts so stratified train_test_split inside the splitter
        # doesn't fail due to classes with only one member in a holdout fold.
        n_pos = 50
        n_neg = 50
        xs = np.arange(n_pos + n_neg)
        labels = np.array([1] * n_pos + [0] * n_neg)
        df = pd.DataFrame({"x": xs, "y": labels})

        splitter = RepeatedKFoldsSplitter(
            n_splits=5, n_repeats=1, shuffle=True, random_state=0, ratio=(8, 1, 1)
        )
        splits = splitter.split(df, stratify_col="y")

        # For each holdout, check that holdout indices (val+test) maintain class proportions reasonably
        overall_pos_frac = labels.mean()
        for s in splits:
            hold_idx = np.concatenate([s.val_idx, s.test_idx])
            hold_labels = labels[hold_idx]
            hold_pos_frac = hold_labels.mean()
            # Because holdout size may vary but is large enough, require close match
            self.assertAlmostEqual(hold_pos_frac, overall_pos_frac, delta=0.15)

    def test_reproducibility_with_random_state(self):
        df = pd.DataFrame({"x": np.arange(20)})
        # Two splitter instances with same random_state should produce identical splits
        splitter_a = RepeatedKFoldsSplitter(
            n_splits=4, n_repeats=2, shuffle=True, random_state=123
        )
        splitter_b = RepeatedKFoldsSplitter(
            n_splits=4, n_repeats=2, shuffle=True, random_state=123
        )

        splits_a = splitter_a.split(df)
        splits_b = splitter_b.split(df)

        self.assertEqual(len(splits_a), len(splits_b))
        for a, b in zip(splits_a, splits_b):
            np.testing.assert_array_equal(a.train_idx, b.train_idx)
            np.testing.assert_array_equal(a.val_idx, b.val_idx)
            np.testing.assert_array_equal(a.test_idx, b.test_idx)


if __name__ == "__main__":
    unittest.main()
