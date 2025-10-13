# test_repeated_group_aware.py
import unittest
import warnings
import numpy as np
import pandas as pd

# adjust path if necessary
from synrxn.split.repeated_group_aware_kfold import RepeatedGroupAwareSplitter


class TestRepeatedGroupAwareSplitter(unittest.TestCase):
    def test_init_invalid_args(self):
        # n_splits < 2
        with self.assertRaises(ValueError):
            RepeatedGroupAwareSplitter(n_splits=1, label_key="g")

        # missing label_key
        with self.assertRaises(ValueError):
            RepeatedGroupAwareSplitter(n_splits=3, label_key=None)

        # invalid holdout_split_mode
        with self.assertRaises(ValueError):
            RepeatedGroupAwareSplitter(
                n_splits=3, label_key="g", holdout_split_mode="bad_mode"
            )

        # ratio sum <= 0
        with self.assertRaises(ValueError):
            RepeatedGroupAwareSplitter(n_splits=3, label_key="g", ratio=(0, 0, 0))

    def test_fit_requires_label_key_present(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        splitter = RepeatedGroupAwareSplitter(n_splits=2, n_repeats=1, label_key="g")
        with self.assertRaises(KeyError):
            splitter.fit(df)

    def test_fit_and_basic_get_split_behavior(self):
        # create a simple dataset with clear groups
        rows = []
        for g in ("A", "B", "C", "D", "E"):
            for i in range(3):  # cluster size = 3 for each label
                rows.append({"value": f"{g}{i}", "group": g})
        df = pd.DataFrame(rows)
        splitter = RepeatedGroupAwareSplitter(
            n_splits=3, n_repeats=2, label_key="group", random_state=42
        )
        splitter.fit(df)

        # get number of generated repeats stored
        self.assertEqual(len(splitter._assignments_per_repeat), splitter.n_repeats)

        # ask for a split (repeat 0, fold 0)
        train_idx, val_idx, test_idx = splitter.get_split(repeat=0, holdout_fold=0)
        # returned indices are pandas.Index into original df
        self.assertIsInstance(train_idx, pd.Index)
        self.assertIsInstance(val_idx, pd.Index)
        self.assertIsInstance(test_idx, pd.Index)

        # union of indices should cover all rows (since val+test+train partition)
        union_index = train_idx.union(val_idx).union(test_idx)
        # original index should equal union
        self.assertEqual(set(union_index.tolist()), set(df.index.tolist()))

        # labels in train/val/test should be disjoint (no group leaks) according to check_group_separation
        ok, results = splitter.check_group_separation()
        self.assertTrue(ok)
        # find the record for repeat=0 fold=0 and confirm 'ok' True
        rec = next(r for r in results if r["repeat"] == 0 and r["fold"] == 0)
        self.assertTrue(rec["ok"])

    def test_get_split_before_fit_raises(self):
        splitter = RepeatedGroupAwareSplitter(n_splits=3, n_repeats=1, label_key="g")
        with self.assertRaises(RuntimeError):
            splitter.get_split(0, 0)
        with self.assertRaises(RuntimeError):
            splitter.get_split_arrays(0, 0)
        with self.assertRaises(RuntimeError):
            list(splitter.split_generator())

    def test_invalid_repeat_or_fold_index_raises(self):
        rows = [{"x": i, "g": i % 3} for i in range(9)]
        df = pd.DataFrame(rows)
        splitter = RepeatedGroupAwareSplitter(
            n_splits=3, n_repeats=1, label_key="g", random_state=1
        )
        splitter.fit(df)
        with self.assertRaises(IndexError):
            splitter.get_split(repeat=10, holdout_fold=0)
        with self.assertRaises(IndexError):
            splitter.get_split(repeat=0, holdout_fold=10)

    def test_warning_when_clusters_less_than_n_splits(self):
        # only 2 labels but n_splits = 5 -> should warn
        rows = [{"x": i, "g": "A" if i < 5 else "B"} for i in range(10)]
        df = pd.DataFrame(rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splitter = RepeatedGroupAwareSplitter(
                n_splits=5, n_repeats=1, label_key="g", random_state=0
            )
            splitter.fit(df)
            # at least one warning should be issued (UserWarning)
            self.assertTrue(any(issubclass(wi.category, UserWarning) for wi in w))

    def test_split_generator_length_and_consistency(self):
        # build dataset with 6 labels
        rows = []
        for g in range(6):
            for i in range(2):
                rows.append({"x": f"{g}-{i}", "lab": f"L{g}"})
        df = pd.DataFrame(rows)
        n_splits = 3
        n_repeats = 4
        splitter = RepeatedGroupAwareSplitter(
            n_splits=n_splits, n_repeats=n_repeats, label_key="lab", random_state=5
        )
        splitter.fit(df)

        gen = list(splitter.split_generator())
        self.assertEqual(len(gen), n_splits * n_repeats)

        # Check that get_split and split_generator yield the same content for first
        tr1, va1, te1 = gen[0]
        tr2, va2, te2 = splitter.get_split(0, 0)
        self.assertTrue(tr1.equals(tr2))
        self.assertTrue(va1.equals(va2))
        self.assertTrue(te1.equals(te2))

    def test_get_split_arrays_and_positions_mapping(self):
        rows = []
        labels = []
        for label in ("X", "Y", "Z"):
            for i in range(4):
                rows.append({"payload": i, "label": label})
                labels.append(label)
        df = pd.DataFrame(rows)
        splitter = RepeatedGroupAwareSplitter(
            n_splits=2, n_repeats=1, label_key="label", random_state=7
        )
        splitter.fit(df)

        pos_train, pos_val, pos_test = splitter.get_split_arrays(0, 0)
        # positions should be integer numpy arrays
        self.assertIsInstance(pos_train, np.ndarray)
        self.assertIsInstance(pos_val, np.ndarray)
        self.assertIsInstance(pos_test, np.ndarray)

        # Back-map to labels and check no label appears in more than one split
        labels_arr = np.array(df.index.map(lambda x: df.loc[x, "label"]))
        train_labels = set(labels_arr[pos_train])
        val_labels = set(labels_arr[pos_val])
        test_labels = set(labels_arr[pos_test])
        self.assertTrue(train_labels.isdisjoint(val_labels))
        self.assertTrue(train_labels.isdisjoint(test_labels))
        self.assertTrue(val_labels.isdisjoint(test_labels))

    def test_cluster_level_holdout_splitting_respects_whole_clusters(self):
        # 4 groups with varying sizes
        rows = []
        for g, size in zip(("A", "B", "C", "D"), (1, 2, 3, 4)):
            for i in range(size):
                rows.append({"v": f"{g}{i}", "grp": g})
        df = pd.DataFrame(rows)
        splitter = RepeatedGroupAwareSplitter(
            n_splits=2,
            n_repeats=1,
            label_key="grp",
            random_state=11,
            holdout_split_mode="cluster_level",
            ratio=(8, 1, 1),
        )
        splitter.fit(df)

        # For each split, ensure val/test sets are whole-cluster granularity (i.e., each label wholly in val or test)
        for rep in range(splitter.n_repeats):
            for fold in range(splitter.n_splits):
                tr_idx, va_idx, te_idx = splitter.get_split(rep, fold)
                # collect labels per set
                tr_labels = set(df.loc[tr_idx, "grp"])
                va_labels = set(df.loc[va_idx, "grp"])
                te_labels = set(df.loc[te_idx, "grp"])
                # they should be disjoint
                self.assertTrue(tr_labels.isdisjoint(va_labels))
                self.assertTrue(tr_labels.isdisjoint(te_labels))
                self.assertTrue(va_labels.isdisjoint(te_labels))

    def test_separate_fold_holdout_selects_whole_fold_as_val(self):
        # make several groups so folds exist
        rows = []
        for g in range(6):
            for i in range(2):
                rows.append({"x": f"{g}-{i}", "group": f"G{g}"})
        df = pd.DataFrame(rows)
        splitter = RepeatedGroupAwareSplitter(
            n_splits=3,
            n_repeats=3,
            label_key="group",
            random_state=3,
            holdout_split_mode="separate_fold",
        )
        splitter.fit(df)

        # for each split ensure val corresponds to whole fold and test to holdout fold
        for rep in range(splitter.n_repeats):
            for holdout_fold in range(splitter.n_splits):
                tr_idx, va_idx, te_idx = splitter.get_split(rep, holdout_fold)
                # candidate folds other than holdout_fold should include val fold's labels wholly
                # build mapping pos->fold from assignments
                assignment = splitter._assignments_per_repeat[rep]
                pos_to_fold = {}
                for cid, cl in enumerate(splitter._clusters):
                    f = assignment[cid]
                    for p in cl:
                        pos_to_fold[p] = f
                # map va_idx positions to fold ids
                va_positions = [splitter._orig_index.get_loc(lbl) for lbl in va_idx]
                va_fold_ids = set(pos_to_fold[p] for p in va_positions)
                # val_fold should be a single fold id (entire fold)
                self.assertEqual(len(va_fold_ids), 1)
                # val_fold_id = next(iter(va_fold_ids))
                # test positions fold should equal holdout_fold
                te_positions = [splitter._orig_index.get_loc(lbl) for lbl in te_idx]
                te_fold_ids = set(pos_to_fold[p] for p in te_positions)
                self.assertEqual(te_fold_ids, {holdout_fold})

    def test_reproducibility_of_assignments_given_random_state(self):
        # reproducibility: two instances with same random_state should produce same splits
        rows = []
        for g in range(12):
            for i in range(2):
                rows.append({"x": f"{g}-{i}", "lab": f"L{g}"})
        df = pd.DataFrame(rows)
        a = RepeatedGroupAwareSplitter(
            n_splits=4, n_repeats=3, label_key="lab", random_state=99
        )
        b = RepeatedGroupAwareSplitter(
            n_splits=4, n_repeats=3, label_key="lab", random_state=99
        )
        a.fit(df)
        b.fit(df)

        # compare get_split_arrays for all repeats/folds
        for rep in range(a.n_repeats):
            for fold in range(a.n_splits):
                a_train, a_val, a_test = a.get_split_arrays(rep, fold)
                b_train, b_val, b_test = b.get_split_arrays(rep, fold)
                np.testing.assert_array_equal(a_train, b_train)
                np.testing.assert_array_equal(a_val, b_val)
                np.testing.assert_array_equal(a_test, b_test)


if __name__ == "__main__":
    unittest.main()
