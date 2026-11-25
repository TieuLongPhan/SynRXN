The new solution-barrier source is the chapter 5 workbook in Lingfeng Gui's
[pinned Zenodo dataset version 16358189](https://doi.org/10.5281/zenodo.16358189),
licensed **CC BY 4.0 in the data record itself**. The LiPRED-2026
[PCCP paper](https://doi.org/10.1039/D6CP00088F) explicitly identifies this workbook
through the concept DOI 10.5281/zenodo.8396100. The accompanying README also
identifies chapter 5 as the complete QM reaction-kinetics data. This does not
rely on transferring the article's separate CC BY 3.0 license to external data.

Reproduce the acquisition and export with:

```sh
python kdd/coverage/adapters/solution_barriers.py --fetch
python kdd/coverage/adapters/solution_barriers.py
python -m unittest kdd.coverage.test_solution_barriers
```

The source lock pins the 36,346,814-byte original ZIP and its extracted
1,034,381-byte `chapter5/QM_dataset.xlsx`, source metadata and README. Only the
workbook, source record, README, attribution and license notices belong to this
table's release dependencies. Other chapters in the acquisition ZIP are not
selected for publication by this adapter.

The prepared table contains **4,514 native cached computed reference values**
from **970 workbook context rows**, across the seven methods named in the paper:

| Method | Native target column | Finite values |
|---|---|---:|
| TC | G_TC | 970 |
| TC-vib | G_TC_VibCorr | 970 |
| TC-SMD | G_TC_AVGof6 | 397 |
| TC-high | G_TC_high | 397 |
| Direct | G_Direct | 413 |
| TC-quasi | quasi_G_TC | 970 |
| Direct-quasi | quasi_G_dir | 397 |

These are solution-phase activation Gibbs free energies, **ΔG‡ in kcal/mol at
298.15 K**, using SMD continuum solvation. They are not Arrhenius activation
energies or gas-phase electronic barriers. The paper defines a solution standard
state of 1 mol/L and gas reference state of 1 atm. The native per-row reference
state correction is retained as provenance and is already included in cached
targets; it is not applied a second time. Each exported target is checked against
both workbook sheets. Cached Excel formulas are read without recalculation.

There are **30 unordered native-name/channel groups** and **542 source solvent
tokens**, including nine hypothetical solvent identifiers. The paper reports
28 reactions and 4,513 values. The broader thesis workbook includes two Fmoc
deprotection groups and yields 4,514 cells for the seven selected methods. The
exact paper subset and the one-value difference have **not been reconciled**;
this export is explicitly the pinned native workbook rather than a claimed exact
reproduction of the article's headline count.

The workbook provides names and codes, rather than reaction or solvent SMILES.
No chemical structures are invented. The 54 references labeled `hypo1` through
`hypo9` lack solvent descriptors in this workbook and are flagged as unsuitable
for evaluation on physical solvent identities. Entries 408 and 531 repeat one
named Diiodomethane/Menschutkin context with different cached targets; all six
associated references are flagged. Both are retained for transparent source
accounting, with no averaging or silent deduplication. The companion sheets
swap reactant order in 22 entries; the unordered grouping reconciles these
without modifying the exported native names.

All solvents, theory levels and method variants of the same named reaction and
channel share `reaction_group`. This prevents treating method or solvent copies
as independent reactions when designing splits. No train/test split or chemical
scaffold independence has been established. Mapped graph models need a separate,
audited identity-resolution step before this names-only source can be used.

Two additional acquisition candidates were verified but are not added by this
adapter:

- [RxnSolvKSE, Chung and Green, Chemical Science 2024](https://doi.org/10.1039/D3SC05353A):
  [actual CC BY 4.0 dataset](https://zenodo.org/records/8423911), with a 311 MB
  `RxnSolvKSE_dataset_v1.1.zip`. It contains reaction/solvent SMILES and computed
  solvation free energies and enthalpies **of activation**, ΔΔGsolv‡ and
  ΔΔHsolv‡. These are solvent corrections rather than absolute barriers.
- [QuantumPioneer solvation data](https://zenodo.org/records/20060606): CC BY 4.0,
  separate reaction-solvation archive across 295 solvents (1.648 GB). Its
  `DDGsolv_forward` is also a solvent contribution to activation, separate from
  the already integrated gas-reference HAT kinetics table. It should not be
  silently relabeled as an absolute solution ΔG‡.
