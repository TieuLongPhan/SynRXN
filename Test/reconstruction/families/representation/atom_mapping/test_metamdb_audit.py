"""Check chemical selection and evidence requirements for MetAMDB references."""
import unittest

from rdkit.Chem import rdChemReactions

from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import inspect_rxn, reference_status


def rxn_block(smiles):
    reaction = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    return rdChemReactions.ReactionToRxnBlock(reaction).encode()


class MetamdbAuditTest(unittest.TestCase):
    def test_concrete_rxn_query_atoms_and_spectators(self):
        raw = rxn_block('[CH3:1][OH:2].[OH2:3]>>[CH2:1]=[O:2].[OH2:3]')
        record, status, _ = inspect_rxn(raw)
        self.assertEqual(status, 'usable_heavy_aam')
        self.assertEqual(record['spectator_pairs_removed'], 1)
        self.assertNotIn(':3', record['candidate_mapping'])
        self.assertEqual(record['rxn'], 'CO>>C=O')
        self.assertFalse(record['element_balanced'])

    def test_nonconserved_mapping_and_generic_atoms(self):
        for smiles, expected in [
            ('[CH3:1][OH:2]>>[CH2:1]=[O:3]', 'heavy_maps_not_conserved'),
            ('[*:1][OH:2]>>[*:1]=[O:2]', 'generic_or_dummy_atom'),
            ('[CH3:1][OH:2]>>[CH3:1][OH:2]', 'unchanged_after_spectator_removal'),
        ]:
            with self.subTest(smiles=smiles):
                record, status, _ = inspect_rxn(rxn_block(smiles))
                self.assertIsNone(record)
                self.assertEqual(status, expected)

    def test_hydrogen_mapping_is_not_claimed(self):
        record, status, _ = inspect_rxn(rxn_block('[CH3:1][O-:2].[H+:3]>>[CH3:1][OH:2]'))
        self.assertEqual(status, 'usable_heavy_aam')
        self.assertTrue(record['hydrogen_maps_differ'])
        self.assertNotIn(':3', record['candidate_mapping'])
        self.assertTrue(record['element_balanced'])
        self.assertTrue(record['charge_balanced'])

    def test_human_reference_needs_review_flag_and_same_payload(self):
        raw = rxn_block('[CH3:1][OH:2]>>[CH2:1]=[O:2]')
        self.assertEqual(reference_status(None, raw)[1], 'curation_not_established')
        self.assertEqual(reference_status({'updated': False}, raw)[1], 'computational_reference')
        self.assertEqual(reference_status({'updated': True}, raw)[1], 'curated_version_not_checked')
        self.assertEqual(reference_status({'updated': True, 'rxnFile': 'other'}, raw)[1],
                         'curated_version_differs')
        self.assertEqual(reference_status({'updated': True, 'rxnFile': raw.decode()}, raw)[1],
                         'upstream_human_curated_reference')


if __name__ == '__main__':
    unittest.main()
