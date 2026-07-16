import pandas as pd

from synkit.Chem.Reaction.Mapper import AAMValidator as SynKitValidator
from synrxn.aam.aam_validator import AAMValidator as SynRXNValidator
from synrxn.metrics.template import acc_aam


def test_synkit_15_matches_synrxn_for_balanced_and_unbalanced_maps():
    row = pd.read_csv("Data/aam/golden.csv.gz", nrows=2).iloc[1]
    legacy = SynRXNValidator()
    current = SynKitValidator(strip_unbalanced_maps=True)
    strict = SynKitValidator(strip_unbalanced_maps=False)

    for method in ("RC", "ITS"):
        expected = legacy.smiles_check(row.rxn_mapper, row.ground_truth, method)
        assert current.smiles_check(row.rxn_mapper, row.ground_truth, method) is expected
        assert strict.smiles_check(row.rxn_mapper, row.ground_truth, method) is not expected


def test_public_aam_metric_uses_synkit_15_response_shape():
    reaction = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
    assert acc_aam([reaction], [reaction]) == 100.0
