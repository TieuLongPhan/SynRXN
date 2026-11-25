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
            # use split_with_val to trigger computation and the size check
            list(splitter.split_with_val(df))

    def test_basic_split_and_get_split(self):
        n = 10
        df = pd.DataFrame({"x": np.arange(n)})
        splitter = RepeatedKFoldsSplitter(
            n_splits=5, n_repeats=1, ratio=(8, 1, 1), shuffle=True, random_state=42
        )
        splits = list(splitter.split_with_val(df))  # returns list[SplitIndices]

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
        # compute splits first
        _ = list(splitter.split_with_val(df))

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
        _ = list(splitter.split_with_val(df))

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
        # stratify series of wrong length -> expect ValueError
        with self.assertRaises(ValueError):
            # use keyword 'stratify' (new API)
            list(splitter.split_with_val(df, stratify=pd.Series(np.arange(5))))

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
        splits = list(splitter.split_with_val(df, stratify="y"))

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

        splits_a = list(splitter_a.split_with_val(df))
        splits_b = list(splitter_b.split_with_val(df))

        self.assertEqual(len(splits_a), len(splits_b))
        for a, b in zip(splits_a, splits_b):
            np.testing.assert_array_equal(a.train_idx, b.train_idx)
            np.testing.assert_array_equal(a.val_idx, b.val_idx)
            np.testing.assert_array_equal(a.test_idx, b.test_idx)

    def test_changed_cohort_recomputes_cached_positions(self):
        splitter = RepeatedKFoldsSplitter(n_splits=3, random_state=7)
        list(splitter.split_with_val(np.zeros((30, 1))))
        splits = list(splitter.split_with_val(np.zeros((18, 1))))
        for split in splits:
            positions = np.concatenate((split.train_idx, split.val_idx, split.test_idx))
            self.assertEqual(sorted(positions.tolist()), list(range(18)))

    def test_sklearn_split_api_recomputes_changed_cohort(self):
        splitter = RepeatedKFoldsSplitter(n_splits=3, random_state=7)
        list(splitter.split(np.zeros((30, 1))))
        for train, holdout in splitter.split(np.zeros((18, 1))):
            self.assertEqual(sorted(np.concatenate((train, holdout)).tolist()), list(range(18)))

    def test_changed_groups_and_labels_recompute_assignments(self):
        frame = pd.DataFrame({"x": np.arange(36)})
        splitter = RepeatedKFoldsSplitter(n_splits=3, random_state=11)
        list(splitter.split_with_val(frame))
        groups = np.array([f"g{i // 3}" for i in range(36)])
        splits = list(splitter.split_with_val(frame, groups=groups))
        for split in splits:
            partitions = [set(groups[index]) for index in
                          (split.train_idx, split.val_idx, split.test_idx)]
            self.assertTrue(partitions[0].isdisjoint(partitions[1]))
            self.assertTrue(partitions[0].isdisjoint(partitions[2]))
            self.assertTrue(partitions[1].isdisjoint(partitions[2]))
        labels = np.array([i % 2 for i in range(36)])
        expected = RepeatedKFoldsSplitter(n_splits=3, random_state=11)
        label_splits = list(splitter.split_with_val(frame, stratify=labels))
        expected_splits = list(expected.split_with_val(frame, stratify=labels))
        for actual, reference in zip(label_splits, expected_splits):
            np.testing.assert_array_equal(actual.train_idx, reference.train_idx)
            np.testing.assert_array_equal(actual.val_idx, reference.val_idx)
            np.testing.assert_array_equal(actual.test_idx, reference.test_idx)

    def test_failed_recomputation_clears_prior_assignments(self):
        splitter = RepeatedKFoldsSplitter(n_splits=3, random_state=2)
        list(splitter.split_with_val(np.zeros((18, 1))))
        with self.assertRaisesRegex(ValueError, "at least two held-out rows"):
            list(splitter.split_with_val(np.zeros((3, 1))))
        self.assertEqual(splitter.n_generated_splits, 0)
        with self.assertRaises(RuntimeError):
            splitter.get_split()

    def test_deterministic_grouped_outer_repeats_are_disclosed(self):
        groups = np.array([f"g{i // 3}" for i in range(36)])
        splitter = RepeatedKFoldsSplitter(n_splits=3, n_repeats=2, random_state=7)
        with self.assertWarnsRegex(UserWarning, "not independent outer splits"):
            splits = list(splitter.split_with_val(np.zeros((36, 1)), groups=groups))
        self.assertEqual(splitter.outer_repeat_policy_, "deterministic_group_outer_folds")
        for first, second in zip(splits[:3], splits[3:]):
            first_holdout = np.sort(np.concatenate((first.val_idx, first.test_idx)))
            second_holdout = np.sort(np.concatenate((second.val_idx, second.test_idx)))
            np.testing.assert_array_equal(first_holdout, second_holdout)

    def test_unshuffled_outer_folds_have_no_random_state_error(self):
        splitter = RepeatedKFoldsSplitter(n_splits=3, shuffle=False, random_state=7)
        splits = list(splitter.split_with_val(np.zeros((18, 1))))
        self.assertEqual(len(splits), 3)
        self.assertEqual(splitter.outer_repeat_policy_, "deterministic_unshuffled_outer_folds")

    def test_lazy_stratification_failure_uses_documented_fallback(self):
        splitter = RepeatedKFoldsSplitter(n_splits=3, random_state=7)
        labels = np.arange(18)
        with self.assertWarnsRegex(UserWarning, "Falling back to KFold"):
            splits = list(splitter.split_with_val(np.zeros((18, 1)), stratify=labels))
        self.assertEqual(len(splits), 3)

    def test_zero_repeats_and_indivisible_holdout_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "n_repeats"):
            RepeatedKFoldsSplitter(n_repeats=0)
        splitter = RepeatedKFoldsSplitter(n_splits=3, ratio=(8, 1, 100), random_state=7)
        with self.assertRaisesRegex(ValueError, "cannot form nonempty validation and test"):
            list(splitter.split_with_val(np.zeros((6, 1))))

    def test_group_kfold_holdout_keeps_groups_and_rejects_indivisible_folds(self):
        splitter = RepeatedKFoldsSplitter(n_splits=3, n_repeats=1, random_state=7)
        groups = np.repeat(np.arange(12), 3)
        features = np.zeros((len(groups), 1))
        for split in splitter.split_with_val(features, groups=groups):
            partitions = split.train_idx, split.val_idx, split.test_idx
            train, validation, test = (set(groups[indices]) for indices in partitions)
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(sorted(np.concatenate(partitions)), list(range(len(groups))))
        with self.assertRaisesRegex(ValueError, "at least two independent groups"):
            list(splitter.split_with_val(np.zeros((6, 1)), groups=np.repeat(np.arange(3), 2)))


if __name__ == "__main__":
    unittest.main()
