# Selectivity source provenance

The provisional selectivity family contains only experimentally observed
targets with explicit repository-level redistribution terms.

| Artifact | Raw task rows | Accepted | Observed labels | License |
| --- | ---: | ---: | ---: | --- |
| `selectivity/hitea_chiral_resolution.csv.gz` | 129 | 50 | 50 signed product-ee measurements | MIT |
| `selectivity/regiosqm_eas.csv.gz` | 535 | 535 | 604 observed EAS sites | MIT |

The HiTEA supplementary methods state that chiral salt resolution products were
measured by chiral SFC-MS and report product percent ee. Seventy-nine rows with
zero in the reported product-yield field are excluded rather than interpreted
as racemic measurements. The sign is preserved exactly, but the source does
not define the favored enantiomer identity row by row.

The Jensen Group repository is a database-only repository: its README calls
the tables a database of real compounds and experimental EAS centers, and the
root repository license is MIT. RegioSQM atom indices are zero-based and tied
to the exact upstream substrate SMILES. Canonical SMILES are identity aids only.

Rebuild both deterministic artifacts with:

```bash
python script/build_selectivity_dataset.py \
  --hitea-input /path/8_SEPT_APPROVED_full_dataset.csv \
  --regiosqm-dir /path/db-regioselectivity
```

The outputs are provisional working-tree artifacts and are not part of the
immutable SynRXN 1.3.0 manifest.
