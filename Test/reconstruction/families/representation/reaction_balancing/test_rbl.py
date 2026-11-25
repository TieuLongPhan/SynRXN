"""Balancing checks include implicit hydrogens and retain discrepancy direction."""
import pytest

from synrxn.reconstruction.families.representation.reaction_balancing._sources.conversion import reaction_missing_side


@pytest.mark.parametrize("reaction,status,difference", [
    ("[CH3:1][OH:2]>>[CH3:9][OH:8]", "balanced", {"C": 0, "H": 0, "O": 0}),
    ("C>>[CH3]", "one-side", {"C": 0, "H": 1}),
    ("C>>N", "both", {"C": 1, "N": -1, "H": 1}),
])
def test_element_inventory_includes_hydrogens(reaction, status, difference):
    assert reaction_missing_side(reaction, return_details=True) == {"status": status, "diff": difference}


def test_invalid_molecule_cannot_be_reported_as_balanced():
    with pytest.raises(ValueError, match="Cannot parse SMILES"):
        reaction_missing_side("invalid>>C")
